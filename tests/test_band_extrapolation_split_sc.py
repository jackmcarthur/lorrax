"""The two 2026-08-16 features MEETING: a χ/Σ split under a default-on SC
band extrapolation.

Neither branch tested this, because on each branch alone it could not happen.
``feat/nband-chi-sigma-split-2026-08-16`` pinned that the brackets come from
the **Σ** count, but on that branch the extrapolation was OFF by default and
its result was reported and discarded.  ``feat/use-band-extrapolation-sc-
2026-08-16`` made the extrapolated Σ_c drive the self-consistent E_nk, but on
that branch there was one band count and "the Σ count" was not a distinct
number.  Merged, the two compose into a claim neither file states:

    **On a split deck, the number that drives self-consistency is derived
    from the Σ count, while the ISDF ζ fit is sized by max(χ, Σ) — both at
    once, from one resolved config.**

Get either half wrong and the failure is silent.  Bracket off the χ count and
the SC loop converges to a fixed point of a curve the run never evaluates.
Size the ISDF off the Σ count and the χ0 sum extrapolates in a ζ basis that
was never fitted for its bands.

WHAT EACH SECTION HOLDS.

  §1  The two halves hold SIMULTANEOUSLY on one resolved split config.
  §2  The SC-driving weights are the Σ-derived counts' weights (item 2).
  §3  The ``nband >= 2*n_occ`` gate is about the Σ count (item 3 ruling).
  §4  The non-PPM guard asks the LADDER, not the stage (item 4 correction).

LAYERING.  §1–§3 are pure: integer resolution, planner arithmetic and OLS
weights, no jax, no WFN, no device.  §4 imports ``gw.sigma_dispatch`` and is
skipped where jax is unavailable.  §2 also reads two production call sites out
of the AST, for the same reason ``test_band_extrapolation_sigma_count.py``
does: no unit fixture in this suite can run a split deck end to end, and a
source-text assertion fails the moment someone reverts a call site even when
nothing in the suite exercises it.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import sys
import types

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")

#: The owner's motivating configuration: χ at the full band count, Σ short and
#: extrapolated.  148 bands apart, so a planner fed the wrong count is off by
#: more than the entire Σ sum and no tolerance can hide it.
CHI, SIGMA, N_OCC = 248, 100, 8


# ---------------------------------------------------------------------------
#  A jax-free handle on gw_config (same stub path as test_band_count_split)
# ---------------------------------------------------------------------------

def _gw_config():
    if "gw.gw_config" in sys.modules:
        return sys.modules["gw.gw_config"]
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    try:
        import jax  # noqa: F401
    except Exception:                                   # noqa: BLE001
        pass
    else:
        from gw import gw_config
        return gw_config
    common = types.ModuleType("common")
    common.__path__ = [os.path.join(_SRC, "common")]
    sys.modules.setdefault("common", common)
    units = types.ModuleType("common.units")
    units.RYD_TO_EV = 13.6056980659
    units.EV_TO_RYD = 1.0 / 13.6056980659
    sys.modules.setdefault("common.units", units)
    gw = types.ModuleType("gw")
    gw.__path__ = [os.path.join(_SRC, "gw")]
    sys.modules.setdefault("gw", gw)
    spec = importlib.util.spec_from_file_location(
        "gw.gw_config", os.path.join(_SRC, "gw", "gw_config.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gw.gw_config"] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_split(chi=CHI, sigma=SIGMA):
    cfg = _gw_config()
    keys = ("number_bands", "number_bands_chi", "number_bands_sigma", "nband")
    params = {k: cfg._DEFAULTS[k] for k in keys}
    params["number_bands_chi"] = chi
    params["number_bands_sigma"] = sigma
    return cfg.resolve_band_counts(
        params, deck_named=("number_bands_chi", "number_bands_sigma"))


def _flat_spectrum(nb, nk=4):
    """Non-degenerate and evenly spaced, so the multiplet snap never moves a
    cut and the counts ARE the fractions."""
    return np.tile(np.linspace(1.0, 20.0, nb), (nk, 1))


# ---------------------------------------------------------------------------
# (1) BOTH HALVES AT ONCE, off ONE resolved config
# ---------------------------------------------------------------------------

def test_one_split_config_sizes_the_isdf_by_the_max_and_brackets_the_sigma():
    """THE INTERACTION NEITHER BRANCH TESTED, in one assertion block.

    The bug this rules out is not "one of the two is wrong" — each branch's
    own suite covers that — it is the two being read off DIFFERENT numbers
    after the merge, which no single-branch test can see.  So the ζ window and
    the bracket plan are derived here from the SAME ``BandCounts`` object, the
    way one run does.
    """
    counts = _resolve_split()

    # -- the ISDF half: sized by the LARGER count, and it says which one --
    assert (counts.chi, counts.sigma) == (CHI, SIGMA)
    assert counts.isdf == max(CHI, SIGMA) == CHI
    assert counts.isdf_source == "chi"
    assert counts.split is True
    assert "number_bands_chi" in counts.describe()

    # -- the bracket half: fractions of the SMALLER (Σ) count --
    from gw.band_extrapolation import BRACKET_FRACTIONS, plan_band_brackets
    plan = plan_band_brackets(
        enabled=True, enk_ry=_flat_spectrum(counts.sigma), n_occ=N_OCC,
        nb_logical=counts.sigma, nb_padded=counts.sigma,
        fractions=BRACKET_FRACTIONS)
    assert plan.counts == (80, 90, 100)
    assert max(plan.counts) == counts.sigma

    # -- and the two are DIFFERENT numbers, which is the whole point --
    assert counts.isdf != max(plan.counts), (
        "on a split deck the ISDF window and the bracket top must not be the "
        "same number; if they are, one of the two halves read the wrong count")
    assert max(plan.counts) < counts.isdf, (
        "every band the Sigma sum brackets must lie inside the loaded window")


def test_the_isdf_window_is_not_sized_by_the_bracket_top():
    """The inverse error, stated separately because it fails differently.

    Sizing the ζ fit by the Σ count on a χ-dominant deck does not crash — χ0
    simply sums 148 bands whose pair densities the interpolation basis was
    never fitted for.  The number that catches it is ``isdf``, not a slice
    bound, so it is asserted against the max rather than against the plan.
    """
    counts = _resolve_split()
    assert counts.isdf == CHI
    assert counts.isdf != SIGMA
    assert min(counts.chi, counts.sigma) not in (counts.isdf,)


@pytest.mark.parametrize("chi,sigma,winner", [
    (248, 100, "chi"),      # the motivating split
    (100, 248, "sigma"),    # the inverse: Sigma the larger consumer
    (100, 100, "tied"),     # unsplit -- the whole tree's decks
])
def test_the_two_halves_stay_consistent_in_both_split_directions(
        chi, sigma, winner):
    """The Σ > χ direction is not symmetric to the χ > Σ one and must be
    checked separately: there the brackets run to the TOP of the loaded
    window, so a call site reading the loaded extent would look correct on
    this deck and be wrong on the other one."""
    from gw.band_extrapolation import plan_band_brackets
    counts = _resolve_split(chi, sigma)
    assert counts.isdf_source == winner
    assert counts.isdf == max(chi, sigma)
    plan = plan_band_brackets(
        enabled=True, enk_ry=_flat_spectrum(counts.sigma), n_occ=N_OCC,
        nb_logical=counts.sigma, nb_padded=counts.sigma)
    assert max(plan.counts) == counts.sigma
    assert max(plan.counts) <= counts.isdf


# ---------------------------------------------------------------------------
# (2) WHAT THE SC LOOP CONSUMES
# ---------------------------------------------------------------------------
# The SC branch's contribution is that S_inf, not S(N3), becomes the E_nk the
# iteration sees.  S_inf is applied to the Sigma cube as a fixed REAL affine
# combination whose coefficients depend ONLY on the three band COUNTS
# (``extrapolation_weights``).  So "which count drives self-consistency" is
# answerable exactly: it is whichever count produced ``plan.counts``.

def test_the_wrong_band_count_is_INVISIBLE_in_the_weights():
    """**The weights cannot detect the substitution, and that is why the call
    site has to be pinned.**

    This test was first written the other way round — asserting that the
    Σ-derived and χ-derived weights are far apart — and it FAILED, which is
    the finding.  ``extrapolation_weights`` is ordinary least squares in
    ``1/N``, and OLS coefficients depend only on the RATIOS of the abscissae.
    The fractions are the same 0.80/0.90/1.00 whichever count they are taken
    of, so the two weight vectors agree to the rounding of the counts:
    ``(80, 90, 100)`` gives ``[-4.295, +0.664, +4.631]`` and
    ``(198, 223, 248)`` gives ``[-4.255, +0.664, +4.591]``.

    So a run that bracketed the wrong count would apply an operator that is
    numerically almost the RIGHT one — to the WRONG three partial sums.  The
    error is entirely in which bands were summed, i.e. in ``plan.bounds``, and
    it is invisible in every weight-level diagnostic.  Nothing downstream can
    catch it; only the call site can.  Hence
    ``test_band_extrapolation_sigma_count.py``'s AST assertions and the ones
    below, and hence this is a documented property rather than a bug.
    """
    from gw.band_extrapolation import (BRACKET_FRACTIONS,
                                       extrapolation_weights,
                                       plan_band_brackets)
    counts = _resolve_split()
    sigma_plan = plan_band_brackets(
        enabled=True, enk_ry=_flat_spectrum(counts.sigma), n_occ=N_OCC,
        nb_logical=counts.sigma, nb_padded=counts.sigma)
    w_sigma = extrapolation_weights(sigma_plan.counts)

    chi_counts = tuple(int(round(f * counts.chi))
                       for f in BRACKET_FRACTIONS) + (counts.chi,)
    w_chi = extrapolation_weights(chi_counts)

    for w in (w_sigma, w_chi):
        assert np.isrealobj(w)
        assert np.isclose(w.sum(), 1.0)

    # Near-identical: the substitution does NOT show up here.
    assert np.allclose(w_sigma, w_chi, atol=0.05), (
        f"the weights are expected to be near scale-invariant; if this ever "
        f"stops being true the docstring above is stale.  {w_sigma} vs "
        f"{w_chi}")

    # Where it DOES show up: the bands actually summed.
    assert sigma_plan.counts == (80, 90, 100)
    assert chi_counts == (198, 223, 248)
    assert max(sigma_plan.counts) == counts.sigma < counts.chi
    assert sigma_plan.bounds[-1][1] == counts.sigma, (
        "the last bracket's upper bound is the only place the count is "
        "recoverable from the plan, so it is the thing to assert")


def test_a_converged_sigma_survives_the_extrapolation_unchanged():
    """``sum(c) == 1`` is not decoration: it is the statement that a Σ which
    does not depend on the band count comes through the SC-driving step
    UNSCALED.  Checked on the Σ-count weights specifically, because that is
    the operator the loop applies."""
    from gw.band_extrapolation import extrapolation_weights, plan_band_brackets
    counts = _resolve_split()
    plan = plan_band_brackets(
        enabled=True, enk_ry=_flat_spectrum(counts.sigma), n_occ=N_OCC,
        nb_logical=counts.sigma, nb_padded=counts.sigma)
    w = extrapolation_weights(plan.counts)
    converged = np.array([1.5 - 0.25j] * 3)
    assert np.isclose(np.tensordot(w, converged, axes=(0, 0)), 1.5 - 0.25j)


# --- the SC seam, read out of the source -----------------------------------

def _calls(path, func_name):
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name == func_name:
            out.append(node)
    return out


def test_the_driving_sigma_is_the_extrapolated_point_of_the_planned_counts():
    """The chain the SC loop rides, pinned end to end in ``ppm_pipeline``:

        plan  <- Sigma-side band fields            (test_..._sigma_count.py)
        cube.band_counts <- plan.counts            (ppm_sigma, below)
        weights <- extrapolation_weights(cube.band_counts)
        sigma_c_body_omega <- _extrapolated_point(cube, weights)   <- drives H

    Any break in it substitutes a different count silently.
    """
    path = os.path.join(_SRC, "gw", "ppm_pipeline.py")
    calls = _calls(path, "_extrapolated_point")
    assert len(calls) == 1, f"expected one driving call, got {len(calls)}"
    args = [ast.unparse(a) for a in calls[0].args]
    assert args[1] == "extrapolation_weights(sigma_omega.band_counts)", args
    # ...and band_counts is the PLAN's counts, not a band count re-read from
    # the config or the meta.
    src = open(os.path.join(_SRC, "gw", "ppm_sigma.py"), encoding="utf-8").read()
    assert "band_counts=tuple(int(c) for c in plan.counts)" in src, (
        "the Sigma cube's band_counts must come from the bracket plan; if it "
        "is re-derived from a config key the split can disagree with it")


def test_the_unextrapolated_twin_exists_only_when_the_feature_drives():
    """The default-off path's object graph must be unchanged, which is what
    keeps the bit-identity claim true.  Read structurally: the twin is
    assigned ``None`` outside the ``plan.enabled`` branch."""
    src = open(os.path.join(_SRC, "gw", "ppm_pipeline.py"),
               encoding="utf-8").read()
    assert "sigma_c_body_omega_unextrap = None" in src
    assert "sigma_c_body_omega_unextrap = sigma_c_body_omega_n3" in src


# ---------------------------------------------------------------------------
# (3) THE GATE IS ABOUT THE Σ COUNT
# ---------------------------------------------------------------------------
# Merge ruling 2026-08-16.  The owner's rule -- "kill the calculation if the
# number of bands requested is not >= 2*N_electrons" -- predates the split, so
# "the number of bands" in it is now ambiguous.  It is the Sigma count.

def test_the_gate_reads_the_sigma_count_even_when_chi_is_enormous():
    """χ = 248 does not rescue a Σ sum of 14 against n_occ = 8.

    A gate that read the loaded extent (``max(chi, sigma)`` = 248) would sail
    through here, and the run would fit a 1/N law to a sum with 6 conduction
    bands.
    """
    from gw.band_extrapolation import (BandExtrapolationRefused,
                                       plan_band_brackets)
    counts = _resolve_split(chi=248, sigma=14)
    assert counts.isdf == 248, "the loaded extent really is the big number"
    with pytest.raises(BandExtrapolationRefused) as exc:
        plan_band_brackets(enabled=True, enk_ry=_flat_spectrum(counts.sigma),
                           n_occ=N_OCC, nb_logical=counts.sigma,
                           nb_padded=counts.sigma)
    msg = str(exc.value)
    assert "number_bands_sigma" in msg
    assert str(2 * N_OCC) in msg, "the floor is 2*n_occ, stated"
    assert "number_bands_chi" in msg and "NOT help" in msg, (
        "the refusal must name the knob that does NOT fix it; an operator "
        "with a 248-band chi count will otherwise raise the wrong one")
    assert "248" not in msg, (
        "the chi count is not in scope for this refusal and quoting it would "
        "suggest it is")


def test_the_gate_passes_on_the_sigma_count_alone():
    """The other side of the same ruling: a Σ count that clears 2*n_occ runs,
    and a small χ beside it is not consulted.  If the gate were read against
    ``min(chi, sigma)`` -- the other plausible misreading -- this would
    refuse."""
    from gw.band_extrapolation import plan_band_brackets
    counts = _resolve_split(chi=10, sigma=40)
    assert counts.isdf == 40 and counts.chi == 10
    plan = plan_band_brackets(
        enabled=True, enk_ry=_flat_spectrum(counts.sigma), n_occ=N_OCC,
        nb_logical=counts.sigma, nb_padded=counts.sigma)
    assert plan.enabled is True
    assert plan.n_occ == N_OCC and plan.n_cond == 40 - N_OCC


def test_the_boundary_is_the_sigma_count_at_exactly_two_n_occ():
    """``number_bands_sigma == 2*n_occ`` RUNS (the 2026-08-16 relaxation),
    stated on a SPLIT deck so the count under test is unambiguous."""
    from gw.band_extrapolation import plan_band_brackets
    counts = _resolve_split(chi=248, sigma=2 * N_OCC)
    plan = plan_band_brackets(
        enabled=True, enk_ry=_flat_spectrum(counts.sigma), n_occ=N_OCC,
        nb_logical=counts.sigma, nb_padded=counts.sigma)
    assert plan.enabled is True
    assert plan.n_cond == plan.n_occ == N_OCC
    assert len(set(plan.counts)) == 3


def test_the_planner_signature_offers_no_chi_argument_at_all():
    """The structural form of the ruling.  The gate cannot be about the χ
    count because the χ count is not among the planner's inputs — stated as a
    test so that adding one becomes a deliberate act with a failing suite."""
    import inspect
    from gw.band_extrapolation import plan_band_brackets
    params = set(inspect.signature(plan_band_brackets).parameters)
    assert not any("chi" in p for p in params), params
    assert {"n_occ", "nb_logical", "nb_padded"} <= params


# ---------------------------------------------------------------------------
# (4) THE NON-PPM GUARD ASKS THE LADDER, NOT THE STAGE
# ---------------------------------------------------------------------------
# The SC branch built this guard on the premise that ``sc_stage_N_type`` "does
# not exist on any branch", concluded from a ``--all`` search in a
# single-branch checkout (where ``--all`` covers only FETCHED refs).  The keys
# are real -- ``origin/feat/staged-sc-2026-08-15`` (98289d77) carries
# ``SC_STAGE_TYPES``, ``SCStage``, ``default_sc_ladder`` and
# ``resolve_sc_stages``.  Read against the real interface:
#
#   * the per-stage DISABLE was accidentally right, because
#     ``run_staged_self_consistency`` rewrites ``compute_mode`` per stage;
#   * the per-stage REFUSAL was wrong, because it kills the run before the
#     stage that would have consumed the key.
#
# These tests are written against the SHAPE of the real interface (an object
# with ``.sc.stages``, each carrying ``.mode``), so they pass on this branch
# and keep passing when staged-SC merges.

def _stage(mode):
    return types.SimpleNamespace(mode=mode)


def _cfg(modes=None, deck_mode=None, explicit=True):
    """A config shaped like the real one, with or without a ladder."""
    sc = None if modes is None else types.SimpleNamespace(
        stages=tuple(_stage(m) for m in modes))
    ns = types.SimpleNamespace(
        sigma=types.SimpleNamespace(band_extrapolation=True,
                                    band_extrapolation_explicit=explicit),
        sc=sc)
    if deck_mode is not None:
        ns.compute_mode = deck_mode
    return ns


def test_sigma_stage_modes_reads_the_ladder_when_there_is_one():
    cfg_mod = _gw_config()
    CM = cfg_mod.ComputeMode
    modes = cfg_mod.sigma_stage_modes(
        _cfg([CM.COHSEX, CM.GN_PPM]), fallback=CM.COHSEX)
    assert modes == (CM.COHSEX, CM.GN_PPM)


def test_sigma_stage_modes_falls_back_to_the_deck_mode_without_a_ladder():
    """Correct BEFORE staged-SC merges, which is the state of this branch."""
    cfg_mod = _gw_config()
    CM = cfg_mod.ComputeMode
    assert cfg_mod.sigma_stage_modes(
        _cfg(None, deck_mode=CM.COHSEX)) == (CM.COHSEX,)
    # ...and to the dispatched mode when the config exposes neither, which is
    # what a hand-made namespace in a unit test looks like.
    assert cfg_mod.sigma_stage_modes(
        _cfg(None), fallback=CM.COHSEX) == (CM.COHSEX,)


def test_the_mpa_default_ladder_is_consumable():
    """``compute_mode = mpa`` takes the ladder ``GN_PPM -> MPA`` (the real
    ``default_sc_ladder``).  A per-stage refusal would kill it at stage 2,
    AFTER paying for a full GN-PPM stage."""
    cfg_mod = _gw_config()
    CM = cfg_mod.ComputeMode
    assert cfg_mod.band_extrapolation_is_consumable((CM.GN_PPM, CM.MPA))
    # MPA consumes the raw cumulative points, not the PPM estimator.
    assert cfg_mod.band_extrapolation_is_consumable((CM.MPA,))
    assert CM.MPA.is_dynamic and CM.MPA.ppm_model is None, (
        "MPA's machinery contract must not pretend it has a PPM pole model")


def test_a_static_only_ladder_is_not_consumable():
    cfg_mod = _gw_config()
    CM = cfg_mod.ComputeMode
    assert not cfg_mod.band_extrapolation_is_consumable(
        (CM.COHSEX, CM.X_ONLY))


# --- the dispatch behaviour itself (needs jax) ------------------------------

def _dispatch(mode, cfg, print_fn=None):
    pytest.importorskip("jax")
    from gw.sigma_dispatch import compute_sigma_xc
    kw = {} if print_fn is None else {"print_fn": print_fn}
    return compute_sigma_xc(
        mode, wfns=None, V_q=None, W_by_role={}, e_qp_ev=None,
        static_head_terms=None, head_resolver=None, quad=None, config=cfg,
        meta=None, mesh_xy=None, sym=None, wfn=None, band_slices=None,
        input_dir=".", **kw)


def test_a_cohsex_first_ladder_stays_runnable_with_an_EXPLICIT_key():
    """**THE BEHAVIOUR THAT MUST SURVIVE.**  ``sc_stage_1_type = cohsex,
    sc_stage_2_type = gnppm`` with the key named: stage 1 must DISABLE and the
    run must continue to stage 2, which is the stage the operator wanted
    extrapolated.  Refusing here would kill the ladder one stage short of its
    own consumer.
    """
    cfg_mod = _gw_config()
    CM = cfg_mod.ComputeMode
    said = []
    # It gets past the guard and dies later on the None operands -- which is
    # the point: the guard did not stop it.
    with pytest.raises(Exception) as exc:
        _dispatch(CM.COHSEX, _cfg([CM.COHSEX, CM.GN_PPM], explicit=True),
                  print_fn=said.append)
    assert not (isinstance(exc.value, NotImplementedError)
                and "use_band_extrapolation" in str(exc.value)), (
        "a COHSEX stage inside a ladder that contains a PPM stage must not "
        "refuse the run")
    blob = "\n".join(said)
    assert "AUTO-DISABLED" in blob
    assert "gn_ppm" in blob, "the note must name the stage that WILL consume it"


def test_a_run_with_no_consuming_stage_still_REFUSES_an_explicit_key():
    """The guard the SC branch was written for is preserved.  An operator who
    names the key on a run in which nothing reads it gets a refusal, not a
    silently-ignored knob."""
    cfg_mod = _gw_config()
    CM = cfg_mod.ComputeMode
    with pytest.raises(NotImplementedError) as exc:
        _dispatch(CM.COHSEX, _cfg([CM.COHSEX, CM.X_ONLY], explicit=True))
    msg = str(exc.value)
    assert "use_band_extrapolation" in msg
    assert "NO stage of this run consumes it" in msg
    assert "cohsex" in msg and "x_only" in msg, (
        "the refusal must show the ladder it inspected, or the operator "
        "cannot tell which stage list was consulted")


def test_an_mpa_stage_is_no_longer_auto_disabled():
    """MPA now consumes bracket machinery, while no estimator is implied."""
    cfg_mod = _gw_config()
    CM = cfg_mod.ComputeMode
    assert cfg_mod.UNIMPLEMENTED_MODES == {}, (
        "this cell exists because MPA stopped being gated at entry; if "
        "something is unimplemented again, say which and why here")
    said = []
    with pytest.raises(BaseException):
        _dispatch(CM.MPA, _cfg([CM.GN_PPM, CM.MPA], explicit=True),
                  print_fn=said.append)
    note = " ".join(said)
    assert "AUTO-DISABLED" not in note, note


def test_mpa_bypasses_the_static_estimator_guard():
    """MPA samples brackets; it must not enter the static 1/N refusal."""
    src = open(os.path.join(_SRC, "gw", "sigma_dispatch.py"),
               encoding="utf-8").read()
    guard = src[src.index("STATIC-MODE ESTIMATOR GUARD"):]
    guard = guard[:guard.index("Static channels")]
    assert "mode is not ComputeMode.MPA" in guard
    assert "estimator=NONE" in src[:src.index("STATIC-MODE ESTIMATOR GUARD")]
    assert 'getattr(mode, "is_dynamic", False)' not in guard
    assert "288.2" in guard, "static refusal must retain its measurement"


def test_a_static_stage_still_carries_the_measurement_that_justifies_it():
    """The converse: on an actually-static stage the note must keep the
    number, because that measurement IS the reason there."""
    cfg_mod = _gw_config()
    CM = cfg_mod.ComputeMode
    said = []
    with pytest.raises(Exception):
        _dispatch(CM.COHSEX, _cfg(None, deck_mode=CM.COHSEX, explicit=False),
                  print_fn=said.append)
    blob = "\n".join(said)
    assert "AUTO-DISABLED" in blob
    assert "288.2" in blob

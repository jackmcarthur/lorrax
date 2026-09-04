"""Gates for the spectrum-resolved shell estimator (``spectral_shell``).

THE LOAD-BEARING ONES ARE THE TWO AT THE TOP.  Everything else here is a
property test on arithmetic; those two are the reason the estimator is
allowed to be the default:

  * :func:`test_reproduces_the_measured_s508_table` — the estimator's error
    against a MEASURED ``S(508)``, a number BerkeleyGW computed, on the
    508-band Si 50 Ry arm.  Held out: no part of it was fitted.  It is
    skipped rather than failed when the arm is not on disk (``$SCRATCH`` is
    purge-eligible), and the skip says so by name.
  * :func:`test_band_index_only_is_the_incumbent_estimator_untouched` — the
    rename changed nothing.  ``band_index_only`` must produce the same
    numbers, from the same code, as the default did before this estimator
    existed.

The rest gate the rulings: β is per-state, the ladder is DFT-only, failure is
a named refusal and never a clip, ``N_T`` is the finite basis, and the
per-state weights leave Σ exactly Hermitian.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from gw.band_extrapolation import (
    BAND_EXTRAPOLATION_ESTIMATORS,
    BAND_EXTRAPOLATION_ESTIMATOR_DEFAULT,
    SHELL_EXPONENT_BRACKET,
    SHELL_FAIL_EDGE,
    SHELL_FAIL_NO_ROOT,
    SHELL_FAIL_SIGN,
    SHELL_FAIL_ZERO,
    SHELL_OK,
    SPECTRAL_EXTRAP_DATASETS,
    SpectralShellExtrapolationFailed,
    build_band_ladder,
    fit_band_extrapolation_spectral,
    format_spectral_report,
    plane_wave_band_count,
    solve_shell_exponents,
    spectral_h5_payload,
    spectral_trust_verdict,
    weyl_ladder_fit,
)
from common.units import RYD_TO_EV


#: The 508-band Si 50 Ry arm.  On ``$SCRATCH``, hence purge-eligible.
S508_RUN = "/pscratch/sd/j/jackm/si_bandtail50_20260816"
S508_PROTO = ("/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/"
              "reports/band_tail_exponent_50ry_2026-08-16/scripts")

#: The published table, ``N_max -> (band_index_only, spectral_shell)`` median
#: |error| against the MEASURED S(508), meV, over the 28 Fermi-window states.
#: Quoted in ``gw.band_extrapolation``'s module docstring and in
#: ``docs/input_reference.md``; this is the only place it is CHECKED.
S508_TABLE = {152: (45.8, 4.7), 204: (29.7, 14.7), 260: (17.5, 12.5),
              296: (12.6, 0.7), 396: (3.5, 0.0)}


# ---------------------------------------------------------------------------
#  fixtures: the prototype binding's import shim, and a synthetic ladder
# ---------------------------------------------------------------------------

def _stub_matplotlib():
    """Register an INERT ``matplotlib`` so the prototype binding imports.

    ``arms.py`` reaches ``make_plots`` for exactly two things -- the degeneracy
    snapper and ``counts_of_total``, the SHIPPED fraction rule -- and that
    module imports ``matplotlib``/``pyplot``/``lines`` AND configures
    ``plt.rcParams`` at file scope.  matplotlib is not in the LORRAX container.

    Stubbed rather than re-derived: re-deriving the snapper would turn this
    into a check of a SECOND implementation of the rungs rather than of the
    shipped one, which is the opposite of the point.  The stub is inert (every
    attribute access, call, index and mutation returns the same do-nothing
    object) BECAUSE the import path configures rcParams -- a stub that raised
    would fail at import rather than at first plot, and there is no plot here
    to reach.  Nothing this script or test asserts depends on matplotlib, so
    inert is safe; anything that DID need a figure would silently get nothing,
    which is why this helper is used only from these two entry points.
    """
    import sys
    import types

    if "matplotlib" in sys.modules:
        return

    class _Inert:
        def __getattr__(self, name):
            return self

        def __call__(self, *a, **k):
            return self

        def __getitem__(self, k):
            return self

        def __setitem__(self, k, v):
            pass

        def update(self, *a, **k):
            pass

        def __iter__(self):
            return iter(())

    _inert = _Inert()

    class _Mod(types.ModuleType):
        def __getattr__(self, name):
            return _inert

    for nm in ("matplotlib", "matplotlib.pyplot", "matplotlib.lines",
               "matplotlib.patches", "matplotlib.colors", "matplotlib.cm",
               "matplotlib.ticker", "matplotlib.gridspec"):
        m = _Mod(nm)
        m.__path__ = []              # a package, so submodule imports resolve
        m.__spec__ = None
        sys.modules[nm] = m
    for nm in ("pyplot", "lines", "patches", "colors", "cm", "ticker",
               "gridspec"):
        object.__setattr__(sys.modules["matplotlib"], nm,
                           sys.modules[f"matplotlib.{nm}"])


def _synthetic_ladder(n_dft=120, nk=4, n_target=400, e0=-6.0, c=3.0, seed=17):
    """A free-electron ladder with a little k dispersion, in Ry.

    Built to the SAME law the estimator fits, so the tests below measure the
    estimator rather than the ladder's ability to describe a real solid.
    """
    rng = np.random.default_rng(seed)
    n = np.arange(1, n_dft + 1, dtype=np.float64)
    base = e0 + c * n ** (2.0 / 3.0)
    enk_ev = base[None, :] + rng.normal(scale=0.05, size=(nk, n_dft))
    enk_ev = np.sort(enk_ev, axis=1)
    return build_band_ladder(enk_ry=enk_ev / RYD_TO_EV,
                             kweights=None, n_target=n_target)


def _points_from_power_law(ladder, counts, beta, amp, base=1.0):
    """``S(N_i)`` for a Σ whose per-band increment is EXACTLY ``A·x^(-β)``.

    The estimator should then recover ``β`` to the bisection's precision and
    predict ``S(N_T)`` exactly, because the model it assumes is the model the
    data was generated from.  ``base`` is the (arbitrary) partial sum below
    ``N₁``: the estimator never sees it and must not depend on it.
    """
    a = [ladder.absolute(int(x)) for x in counts]
    s1 = base
    s2 = s1 - amp * float(ladder.moment(a[0], a[1], beta))
    s3 = s2 - amp * float(ladder.moment(a[1], a[2], beta))
    truth = s3 - amp * float(ladder.moment(a[2], ladder.n_target, beta))
    return np.array([s1, s2, s3]), truth


# ---------------------------------------------------------------------------
#  (1)  THE HELD-OUT TEST  — the reason this estimator is the default
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.path.exists(f"{S508_RUN}/armA/ch_converge.dat")
    or not os.path.exists(f"{S508_PROTO}/arms.py"),
    reason=(f"the 508-band Si 50 Ry arm is not on disk ({S508_RUN}); it lives "
            f"on a PURGE-ELIGIBLE $SCRATCH filesystem, so its absence is an "
            f"ABSENCE OF DATA and not a measurement.  The estimator's "
            f"held-out numbers are reproduced by "
            f"reports/spectral_shell_band_extrapolation_2026-08-17/scripts/"
            f"reference_table.py, which reads the same arm."))
def test_reproduces_the_measured_s508_table():
    """Both estimators, scored against a number BerkeleyGW computed.

    ``S(508)`` is a MEASURED partial sum, not a fit and not a model, so an
    estimator predicting it from ``N_max < 508`` is being graded on data it
    never saw.  The rungs, the Fermi window and the degeneracy-snapped band
    counts all come from the 2026-08-16 study's own binding, so this is the
    same protocol as the prototype and not a re-derivation of it.

    ⚠ THE ERRORS ARE NON-MONOTONE IN ``N_max`` (4.7 at 152 against 14.7 at
    204).  That is the method's own behaviour — it is a two-shell local fit —
    and the table is pinned INCLUDING the non-monotonicity so that nobody
    "fixes" it.
    """
    import sys

    _stub_matplotlib()
    sys.path.insert(0, S508_PROTO)
    try:
        from arms import Arm
        from predict508 import fit_model
    finally:
        sys.path.remove(S508_PROTO)
    import h5py

    arm = Arm("A", f"{S508_RUN}/armA/ch_converge.dat",
              f"{S508_RUN}/armA/sigma_hp.log", f"{S508_RUN}/qe/WFN.h5", "")
    with h5py.File(f"{S508_RUN}/qe/WFN.h5", "r") as f:
        kw = np.asarray(f["mf_header/kpoints/w"][:], float)

    # DFT-ONLY.  ``arm.EL`` is mf_header/kpoints/el — the mean field.
    ladder = build_band_ladder(enk_ry=arm.EL, kweights=kw, n_target=508)
    assert ladder.n0 == 0.0, "the Si deck's Weyl ladder fits with n0 = 0"
    assert ladder.r2 > 0.999, f"Weyl R^2 = {ladder.r2}"

    keys = arm.FERMI
    for nmax, (want_1n, want_sp) in zip((150, 200, 250, 300, 400),
                                        S508_TABLE.values()):
        n = arm.rung(nmax)["n"]
        counts = arm.counts_of_total(n, (0.80, 0.90))
        S = np.empty((3, len(keys)))
        truth = np.empty(len(keys))
        for j, k in enumerate(keys):
            N_int, s, _ = arm.CUR[k]
            g = {int(q): s[i] for i, q in enumerate(N_int)}
            truth[j] = g[508]
            S[:, j] = [g[int(c)] for c in counts]

        fit = fit_band_extrapolation_spectral(counts, S, ladder)
        assert fit.n_failed == 0, fit.failure_report()
        got_sp = float(np.median(np.abs(np.real(fit.s_inf) - truth))) * 1e3
        got_1n = float(np.median([
            abs(fit_model(counts, S[:, j], 1.0) - truth[j]) * 1e3
            for j in range(len(keys))]))
        assert round(got_sp, 1) == want_sp, (
            f"N_max {n}: spectral_shell median |err| {got_sp:.3f} meV, "
            f"published {want_sp}")
        assert round(got_1n, 1) == want_1n, (
            f"N_max {n}: band_index_only median |err| {got_1n:.3f} meV, "
            f"published {want_1n}")
        b = np.asarray(fit.beta)
        assert 3.0 < float(np.median(b)) < 5.5, (
            f"beta median {np.median(b)} outside the measured 3.4-5.3 band; "
            f"beta is the matrix-element falloff a+1 and a drifts 1.83->3.94")


# ---------------------------------------------------------------------------
#  (2)  THE RENAME CHANGED NOTHING
# ---------------------------------------------------------------------------

def test_band_index_only_is_the_incumbent_estimator_untouched():
    """``band_index_only`` IS ``fit_band_extrapolation``, not a copy of it.

    The rename must not have forked the code.  Selecting it must reach the
    same function, produce the same intercept, and produce the SAME three
    scalar weights the previous default applied to the Σ cube — which is what
    makes the byte-identity claim on a real deck a claim about arithmetic
    rather than about luck.
    """
    from gw.band_extrapolation import (
        extrapolation_weights, fit_band_extrapolation)

    assert BAND_EXTRAPOLATION_ESTIMATORS == (
        "spectral_shell", "band_index_only")
    assert BAND_EXTRAPOLATION_ESTIMATOR_DEFAULT == "spectral_shell"

    rng = np.random.default_rng(2026)
    counts = (100, 112, 124)
    S = rng.normal(size=(3, 5, 7)) + 1j * rng.normal(size=(3, 5, 7))
    fit = fit_band_extrapolation(counts, S)
    w = extrapolation_weights(counts)
    assert w.shape == (3,), "the incumbent's weights are three SCALARS"
    assert w.dtype == np.float64
    # The weights ARE the fit: same operator, two entry points.
    assert np.allclose(np.tensordot(w, S, axes=(0, 0)), fit.s_inf, atol=0,
                       rtol=1e-13)


def test_the_deck_key_defaults_to_spectral_shell_and_refuses_a_typo():
    from gw.gw_config import DynamicSigmaConfig

    kw = dict(omega_min_ev=-5.0, omega_max_ev=5.0, omega_step_ev=0.1,
              regularization_ev=0.1, window_edge_factor=1.0,
              fermi_reference="midgap",
              sigma_at_dft_extrapolate=False, sigma_at_dft_energies=False)
    assert DynamicSigmaConfig(**kw).band_extrapolation_estimator == \
        "spectral_shell"
    for name in BAND_EXTRAPOLATION_ESTIMATORS:
        assert DynamicSigmaConfig(
            band_extrapolation_estimator=name,
            **kw).band_extrapolation_estimator == name
    # A misspelling must REFUSE, not fall back to the default: a knob that
    # silently ran the other arm is how a green A/B comes to measure nothing.
    with pytest.raises(ValueError) as exc:
        DynamicSigmaConfig(band_extrapolation_estimator="spectral", **kw)
    for name in BAND_EXTRAPOLATION_ESTIMATORS:
        assert name in str(exc.value)


def test_bracket_scheme_defaults_compatibly_and_refuses_ignored_or_bad_values():
    from gw.band_extrapolation import BRACKET_SCHEMES, BRACKET_SCHEME_DEFAULT
    from gw.gw_config import DynamicSigmaConfig

    kw = dict(omega_min_ev=-5.0, omega_max_ev=5.0, omega_step_ev=0.1,
              regularization_ev=0.1, window_edge_factor=1.0,
              fermi_reference="midgap",
              sigma_at_dft_extrapolate=False, sigma_at_dft_energies=False)
    assert DynamicSigmaConfig(
        **kw).band_extrapolation_bracket_scheme == BRACKET_SCHEME_DEFAULT
    for name in BRACKET_SCHEMES:
        assert DynamicSigmaConfig(
            **kw, band_extrapolation_bracket_scheme=name,
        ).band_extrapolation_bracket_scheme == name
    with pytest.raises(ValueError, match="band_extrapolation_bracket_scheme"):
        DynamicSigmaConfig(**kw, band_extrapolation_bracket_scheme="energy")
    with pytest.raises(ValueError, match="no bracket planner would consume"):
        DynamicSigmaConfig(
            **kw, band_extrapolation=False,
            band_extrapolation_bracket_scheme="conduction_energy_midpoint",
            band_extrapolation_bracket_scheme_explicit=True)


def test_bracket_scheme_deck_key_is_normalized_and_recorded_explicit(tmp_path):
    from gw.gw_config import read_lorrax_input

    deck = tmp_path / "scheme.in"
    deck.write_text(
        "[cohsex]\n"
        "band_extrapolation_bracket_scheme = Conduction_Energy_Midpoint\n")
    params = read_lorrax_input(str(deck))
    assert params["band_extrapolation_bracket_scheme"] == \
        "conduction_energy_midpoint"
    assert "band_extrapolation_bracket_scheme" in params["_deck_named_keys"]


# ---------------------------------------------------------------------------
#  the estimator recovers the law it assumes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("beta", [1.5, 3.0, 4.7, 8.0])
def test_recovers_an_exact_power_law_and_its_tail(beta):
    """Generated from ``A·x^(-β)``, the estimator returns β and S(N_T) exactly.

    Both halves matter.  Recovering β says the root find inverts the moment
    ratio; recovering ``S(N_T)`` says the tail integral uses the SAME β and
    the same moments, so the two are not merely individually plausible.
    """
    lad = _synthetic_ladder()
    counts = (80, 100, 120)
    S, truth = _points_from_power_law(lad, counts, beta, amp=1e-3)
    fit = fit_band_extrapolation_spectral(counts, S[:, None], lad)
    assert int(np.asarray(fit.failure)[0]) == SHELL_OK
    assert abs(float(fit.beta[0]) - beta) < 1e-9, "beta must be recovered"
    assert abs(float(fit.s_inf[0]) - truth) < 1e-12 * max(abs(truth), 1.0)


def test_the_amplitude_and_the_intercept_are_eliminated_analytically():
    """Neither A nor the partial sum below N₁ may reach the answer.

    This is the property that lets three points carry a one-parameter fit.
    Scaling A rescales all three increments together, so β is untouched and
    the CORRECTION scales with it; shifting the intercept moves all three
    points by a constant and must move Ŝ by exactly that constant and change
    nothing else.
    """
    lad = _synthetic_ladder()
    counts = (80, 100, 120)
    S, _ = _points_from_power_law(lad, counts, 4.0, amp=1e-3, base=1.0)
    S10, _ = _points_from_power_law(lad, counts, 4.0, amp=1e-2, base=1.0)
    Sshift, _ = _points_from_power_law(lad, counts, 4.0, amp=1e-3, base=7.5)

    f = fit_band_extrapolation_spectral(counts, S[:, None], lad)
    f10 = fit_band_extrapolation_spectral(counts, S10[:, None], lad)
    fsh = fit_band_extrapolation_spectral(counts, Sshift[:, None], lad)

    assert abs(float(f10.beta[0]) - float(f.beta[0])) < 1e-9, \
        "A cancels from the ratio that determines beta"
    assert abs(float(fsh.beta[0]) - float(f.beta[0])) < 1e-12, \
        "the intercept never enters"
    corr = float(f.s_inf[0] - f.s_at_counts[2, 0])
    assert abs(float(f10.s_inf[0] - f10.s_at_counts[2, 0]) / corr - 10.0) < 1e-8


def test_estar_cancels_from_every_ratio():
    """``E*`` is conditioning, not physics: any positive value gives the same Ŝ.

    Asserted rather than argued because it is the licence to pick ``E*`` by a
    deck-independent rule.  The moments themselves change by ``E*^β``; the
    two ratios the estimator forms do not.
    """
    lad_a = _synthetic_ladder()
    lad_b = build_band_ladder(
        enk_ry=lad_a.e_dft_ev.T / RYD_TO_EV, kweights=lad_a.w_k,
        n_target=lad_a.n_target, estar_window=(3, 9))
    assert abs(lad_a.estar_ev - lad_b.estar_ev) > 1.0, \
        "the two windows must actually give different E*"
    counts = (80, 100, 120)
    S, _ = _points_from_power_law(lad_a, counts, 3.3, amp=1e-3)
    fa = fit_band_extrapolation_spectral(counts, S[:, None], lad_a)
    fb = fit_band_extrapolation_spectral(counts, S[:, None], lad_b)
    assert abs(float(fa.beta[0]) - float(fb.beta[0])) < 1e-9
    assert abs(float(fa.s_inf[0]) - float(fb.s_inf[0])) < 1e-12


# ---------------------------------------------------------------------------
#  beta is per-state, and that is the point
# ---------------------------------------------------------------------------

def test_beta_is_per_state_and_is_never_pooled():
    """Two states with different decays get two different exponents.

    THE OWNER'S RULING, PINNED.  A pooled β would return the median to both
    states and destroy exactly the resolution the estimator exists to give.
    The test constructs states that genuinely differ and asserts that both
    the exponents AND the applied corrections separate.
    """
    lad = _synthetic_ladder()
    counts = (80, 100, 120)
    cols, truths = [], []
    for beta in (2.0, 6.0):
        S, truth = _points_from_power_law(lad, counts, beta, amp=1e-3)
        cols.append(S)
        truths.append(truth)
    S = np.stack(cols, axis=1)                       # (3, 2)
    fit = fit_band_extrapolation_spectral(counts, S, lad)
    assert abs(float(fit.beta[0]) - 2.0) < 1e-9
    assert abs(float(fit.beta[1]) - 6.0) < 1e-9
    for j, truth in enumerate(truths):
        assert abs(float(fit.s_inf[j]) - truth) < 1e-12 * max(abs(truth), 1.0)
    # A pooled exponent -- the median applied to both -- would be wrong on
    # BOTH states.  Quantify it so the ruling has a number behind it.
    pooled = float(np.median(fit.beta))
    r = float(lad.moment(*fit.shells[2], pooled)
              / lad.moment(*fit.shells[1], pooled))
    for j, truth in enumerate(truths):
        s_pool = float(S[2, j] + (S[2, j] - S[1, j]) * r)
        assert abs(s_pool - truth) > 10.0 * abs(float(fit.s_inf[j]) - truth) \
            or abs(s_pool - truth) > 1e-9


# ---------------------------------------------------------------------------
#  failure is a named refusal, never a clip
# ---------------------------------------------------------------------------

def test_sign_change_between_shells_is_a_named_failure():
    lad = _synthetic_ladder()
    counts = (80, 100, 120)
    S = np.array([[0.0], [-1.0], [-0.5]])            # D2 < 0, D3 > 0
    fit = fit_band_extrapolation_spectral(counts, S, lad)
    assert int(np.asarray(fit.failure)[0]) == SHELL_FAIL_SIGN
    assert not np.isfinite(float(np.real(fit.s_inf[0]))), \
        "no value is substituted for a failed state"
    msg = fit.failure_report()
    assert "OPPOSITE SIGN" in msg and "D2" in msg and "D3" in msg
    assert "band_index_only" in msg, "the message names the way forward"


def test_zero_increment_is_a_named_failure():
    lad = _synthetic_ladder()
    fit = fit_band_extrapolation_spectral(
        (80, 100, 120), np.array([[0.0], [0.0], [-1.0]]), lad)
    assert int(np.asarray(fit.failure)[0]) == SHELL_FAIL_ZERO
    assert not np.isfinite(float(np.real(fit.s_inf[0])))


def test_an_unreachable_shell_ratio_refuses_instead_of_clipping():
    """A ratio no power law can produce must FAIL, not return a bracket edge.

    ``g(β) = log I₃ − log I₂`` is strictly decreasing and therefore bounded
    by its own values at the bracket ends.  A ``|D₃/D₂|`` outside
    ``[exp g(hi), exp g(lo)]`` has no root at all, and the failure mode being
    gated is returning ``β = 0.05`` or ``β = 40`` as though it were a fit.
    """
    lad = _synthetic_ladder()
    counts = (80, 100, 120)
    lo, hi = SHELL_EXPONENT_BRACKET
    a1, a2, a3 = (lad.absolute(c) for c in counts)
    g_lo = float(np.log(lad.moment(a2, a3, lo) / lad.moment(a1, a2, lo)))
    g_hi = float(np.log(lad.moment(a2, a3, hi) / lad.moment(a1, a2, hi)))
    assert g_hi < g_lo, "g must be strictly decreasing in beta"

    for ratio in (np.exp(g_lo) * 1e3, np.exp(g_hi) * 1e-3):
        # D2 = -1 so D3 = -ratio reproduces |D3/D2| = ratio with one sign.
        S = np.array([[0.0], [-1.0], [-1.0 - ratio]])
        fit = fit_band_extrapolation_spectral(counts, S, lad)
        code = int(np.asarray(fit.failure)[0])
        assert code in (SHELL_FAIL_NO_ROOT, SHELL_FAIL_EDGE), code
        assert not np.isfinite(float(fit.beta[0])), \
            "a clipped exponent is the exact failure this refuses"
        assert not np.isfinite(float(np.real(fit.s_inf[0])))
        assert "CLIP" in fit.failure_report() or "clip" in \
            fit.failure_report()


def test_a_failed_state_refuses_the_weights_rather_than_emitting_one():
    """One bad state poisons the whole combination, by design.

    The alternative — emitting a coefficient for the states that solved and
    something else for the one that did not — is the per-state fallback the
    owner's ruling forbids: it would ship a Σ assembled from two estimators
    with nothing in the artifact recording which came from which.
    """
    lad = _synthetic_ladder()
    counts = (80, 100, 120)
    good, _ = _points_from_power_law(lad, counts, 3.5, amp=1e-3)
    bad = np.array([0.0, -1.0, -0.5])                      # D2 < 0, D3 > 0
    S = np.stack([bad, good], axis=1)
    fit = fit_band_extrapolation_spectral(counts, S, lad)
    assert fit.n_failed == 1
    assert int(np.asarray(fit.failure)[1]) == SHELL_OK
    with pytest.raises(SpectralShellExtrapolationFailed) as exc:
        fit.weights()
    assert "FAILED on 1 of 2" in str(exc.value)


def test_solve_returns_nan_and_a_code_never_a_bracket_edge():
    """The solver's own contract, independent of the fit that wraps it."""
    lad = _synthetic_ladder()
    a1, a2, a3 = (lad.absolute(c) for c in (80, 100, 120))
    ratio = np.array([1e-30, 1e30, np.nan, 0.0, -1.0])
    beta, code = solve_shell_exponents(lad, (a1, a2), (a2, a3), ratio)
    assert np.all(~np.isfinite(beta)), \
        "every one of these is unreachable; none may come back as a number"
    assert np.all(code != SHELL_OK)


# ---------------------------------------------------------------------------
#  the ladder is DFT-only, and N_T is the finite basis
# ---------------------------------------------------------------------------

def test_the_weyl_ladder_is_fitted_to_the_eigenvalues_alone():
    """E₀, n₀ and C come back from a ladder built to the law, exactly."""
    n = np.arange(1, 301, dtype=np.float64)
    e = -4.25 + 2.75 * (n + 7.0) ** (2.0 / 3.0)
    e0, n0, c, r2 = weyl_ladder_fit(e, 30, 300)
    assert n0 == 7.0
    assert abs(e0 + 4.25) < 1e-8 and abs(c - 2.75) < 1e-10
    assert r2 > 1.0 - 1e-12


def test_n_target_is_the_finite_basis_not_infinity():
    """``N_PW = min(ngk)·nspinor``, and the tail stops there.

    ``S(∞)`` names no physical quantity: the band sum is EXACTLY complete at
    the basis dimension.  ``min`` over k because ngk varies by k-point and
    the minimum is where no k-point is still short.
    """
    assert plane_wave_band_count([1639, 1618, 1604, 1628], 2) == 3208
    assert plane_wave_band_count(np.array([100]), 1) == 100

    lad = _synthetic_ladder(n_dft=120, n_target=400)
    assert lad.n_target == 400
    assert lad.e_weyl_ev.shape == (280,), \
        "bands 121..400 come from the Weyl continuation"
    # The continuation is the SAME law, evaluated one band past the data.
    nxt = lad.e0_ev + lad.c_ev * (121.0 + lad.n0) ** (2.0 / 3.0)
    assert abs(float(lad.e_weyl_ev[0]) - nxt) < 1e-12
    # And it is strictly above the last measured band, i.e. a continuation
    # rather than a restart.
    assert float(lad.e_weyl_ev[0]) > float(lad.e_dft_ev[-1].max())


def test_a_band_sum_already_at_the_basis_refuses():
    """Nothing left to extrapolate is a refusal, not a zero correction."""
    lad = _synthetic_ladder(n_dft=120, n_target=120)
    from gw.band_extrapolation import BandExtrapolationRefused
    with pytest.raises(BandExtrapolationRefused) as exc:
        fit_band_extrapolation_spectral(
            (80, 100, 120), np.zeros((3, 2)), lad)
    assert "complete" in str(exc.value)


def test_the_moment_is_the_weighted_spectral_sum_it_claims_to_be():
    """``log_moment`` against the literal definition, on both segments."""
    lad = _synthetic_ladder(n_dft=40, nk=3, n_target=60)
    for beta in (0.7, 3.0, 11.0):
        for lo, hi in ((5, 20), (30, 40), (35, 55), (40, 60)):
            direct = 0.0
            for i in range(lo, min(hi, lad.n_dft)):
                x = (lad.e_dft_ev[i] - lad.e0_ev) / lad.estar_ev
                direct += float(np.sum(lad.w_k * x ** (-beta)))
            for i in range(max(lo, lad.n_dft), hi):
                x = ((lad.e_weyl_ev[i - lad.n_dft] - lad.e0_ev)
                     / lad.estar_ev)
                direct += float(x ** (-beta))
            got = float(lad.moment(lo, hi, beta))
            assert abs(got - direct) <= 1e-11 * abs(direct)


def test_the_ladder_never_sees_the_self_energy():
    """A structural gate on the DFT-only ruling.

    :func:`build_band_ladder`'s signature is the whole surface through which
    the ladder is constructed, and it takes eigenvalues, weights, an endpoint
    and an offset — no Σ, no S(N_i), nothing that could carry one.  Pinned
    because "E₀ comes from the DFT eigenvalues only" is a ruling that a later
    convenience argument could quietly undo.
    """
    import inspect
    params = set(inspect.signature(build_band_ladder).parameters)
    assert params == {"enk_ry", "kweights", "n_target", "b0", "fit_window",
                      "estar_window"}


# ---------------------------------------------------------------------------
#  the per-state weights, and Hermiticity
# ---------------------------------------------------------------------------

def test_weights_are_real_affine_and_reproduce_the_fit():
    lad = _synthetic_ladder()
    rng = np.random.default_rng(5)
    counts = (80, 100, 120)
    S = np.stack([_points_from_power_law(lad, counts, b, amp=1e-3)[0]
                  for b in rng.uniform(2.0, 6.0, size=(4, 3)).ravel()[:6]],
                 axis=1)
    fit = fit_band_extrapolation_spectral(counts, S, lad)
    w = fit.weights()
    assert w.dtype == np.float64, "complex weights would break Hermiticity"
    assert w.shape == (3,) + fit.s_inf.shape
    assert np.allclose(w.sum(axis=0), 1.0, atol=0, rtol=1e-13), \
        "an affine combination: a band-converged Sigma comes through unchanged"
    assert np.allclose(np.sum(w * S, axis=0), np.real(fit.s_inf),
                       atol=0, rtol=1e-12)


def test_extrapolated_sigma_is_hermitian_to_machine_precision_per_state():
    """The per-state weights must not cost the Hermiticity the scalars had.

    The symmetrisation ``½(w_i + w_j)`` is what buys this, and BITWISE
    equality is asserted rather than ``allclose``: anything less would pass on
    a rule that was only nearly symmetric, and a Σ that is only nearly
    Hermitian gives the next SC iteration eigenvectors inconsistent with its
    own eigenvalues.
    """
    from gw.ppm_pipeline import _extrapolated_point

    rng = np.random.default_rng(817)
    nom, nk, nb = 2, 3, 6
    pts = []
    for _ in range(3):
        A = (rng.normal(size=(nom, nk, nb, nb))
             + 1j * rng.normal(size=(nom, nk, nb, nb)))
        # Hermitian FIRST (this makes the diagonal real), then rebuilt from
        # its own lower triangle so the input is Hermitian to the LAST BIT
        # and the test measures the combination rather than the input.
        A = 0.5 * (A + np.conj(np.swapaxes(A, -1, -2)))
        H = np.tril(A) + np.conj(np.swapaxes(np.tril(A, -1), -1, -2))
        assert np.array_equal(H, np.conj(np.swapaxes(H, -1, -2)))
        pts.append(H)
    cube = np.stack(pts)

    r = rng.uniform(0.1, 5.0, size=(nk, nb))
    w = np.stack([np.zeros_like(r), -r, 1.0 + r], axis=0)
    out = np.asarray(_extrapolated_point(cube, w))
    assert out.shape == (nom, nk, nb, nb)
    assert np.array_equal(out, np.conj(np.swapaxes(out, -1, -2))), \
        "the extrapolated Sigma must be Hermitian to the LAST BIT"
    # And the diagonal must be EXACTLY the per-state estimator, since
    # 1/2(w_i + w_i) = w_i.
    for k in range(nk):
        for i in range(nb):
            want = sum(w[b, k, i] * cube[b, :, k, i, i] for b in range(3))
            assert np.array_equal(out[:, k, i, i], want)


def test_pad_bands_stay_exactly_zero_under_per_state_weights():
    from gw.ppm_pipeline import _extrapolated_point
    nk, nb = 2, 4
    cube = np.zeros((3, 1, nk, nb, nb), dtype=np.complex128)
    cube[:, :, :, :2, :2] = 1.0 + 0.5j
    r = np.linspace(0.5, 3.0, nk * nb).reshape(nk, nb)
    w = np.stack([np.zeros_like(r), -r, 1.0 + r], axis=0)
    out = np.asarray(_extrapolated_point(cube, w))
    assert np.array_equal(out[:, :, 2:, :], np.zeros_like(out[:, :, 2:, :]))


def test_extrapolated_point_refuses_a_weight_shape_it_cannot_mean():
    from gw.ppm_pipeline import _extrapolated_point
    with pytest.raises(ValueError) as exc:
        _extrapolated_point(np.zeros((3, 2, 2)), np.zeros((3, 2)))
    assert "spectral_shell" in str(exc.value)


# ---------------------------------------------------------------------------
#  the report and the h5 payload
# ---------------------------------------------------------------------------

def test_report_carries_the_shells_the_ladder_and_the_per_state_numbers():
    lad = _synthetic_ladder()
    counts = (80, 100, 120)
    S = np.stack([_points_from_power_law(lad, counts, b, amp=1e-3)[0]
                  for b in (2.5, 4.5)], axis=1)[:, :, None]   # (3, 2, 1)
    fit = fit_band_extrapolation_spectral(counts, S, lad)
    from gw.band_extrapolation import plan_band_brackets
    rng = np.random.default_rng(3)
    enk = np.sort(rng.uniform(-1.0, 3.0, size=(2, 120)), axis=1)
    plan = plan_band_brackets(enabled=True, enk_ry=enk, n_occ=20,
                              nb_logical=120, nb_padded=120)
    fit = fit_band_extrapolation_spectral(plan.counts, S, lad)
    text = format_spectral_report(plan, fit, states=[("VBM", (0, 0))])
    for want in ("spectral_shell", "beta", "I_tail/I3", "S_hat", "shells",
                 "N_T", "E0", "PER STATE", "D2", "D3"):
        assert want in text, f"missing {want!r} from the log block"
    # The 1/N block's diagnostics must be absent AND their absence explained.
    assert "Delta_model" in text and "absence is not an omission" in text


def test_h5_payload_names_the_estimator_and_carries_the_ladder():
    lad = _synthetic_ladder()
    from gw.band_extrapolation import plan_band_brackets
    rng = np.random.default_rng(4)
    enk = np.sort(rng.uniform(-1.0, 3.0, size=(2, 120)), axis=1)
    plan = plan_band_brackets(enabled=True, enk_ry=enk, n_occ=20,
                              nb_logical=120, nb_padded=120)
    S = np.stack([_points_from_power_law(lad, plan.counts, b, amp=1e-3)[0]
                  for b in (2.5, 4.5)], axis=1)[:, :, None]
    fit = fit_band_extrapolation_spectral(plan.counts, S, lad)
    pay = spectral_h5_payload(plan, fit)
    assert set(pay["arrays"]) == set(SPECTRAL_EXTRAP_DATASETS)
    assert "sigma_c_extrap_beta_kn" in pay["arrays"]
    assert "sigma_c_extrap_ampl_kn_ev" not in pay["arrays"], \
        "beta must not be written under the 1/N amplitude's name"
    at = pay["attrs"]
    assert at["band_extrapolation_estimator"] == "spectral_shell"
    for key in ("ladder_e0_ev", "ladder_n0", "ladder_estar_ev",
                "ladder_n_target", "shell_bands_absolute", "verdict"):
        assert key in at
    assert at["ladder_n_target"] == lad.n_target


def test_every_spectral_dataset_is_registered_for_star_extraction():
    """A dataset the writer does not know the k axis of is written wrong."""
    from file_io.sigma_output import SIGMA_K_AXIS
    for name in SPECTRAL_EXTRAP_DATASETS:
        assert name in SIGMA_K_AXIS, (
            f"{name} is not in SIGMA_K_AXIS, so its k axis is unknown to the "
            f"star extraction and it would be written unextracted")
        assert SIGMA_K_AXIS[name] == 0


def test_the_verdict_reports_the_spread_and_does_not_claim_quality():
    lad = _synthetic_ladder()
    counts = (80, 100, 120)
    S = np.stack([_points_from_power_law(lad, counts, b, amp=1e-3)[0]
                  for b in (2.0, 3.0, 5.0)], axis=1)
    fit = fit_band_extrapolation_spectral(counts, S, lad)
    v = spectral_trust_verdict(fit)
    assert "beta median" in v and "not a quality metric" in v
    bad = fit_band_extrapolation_spectral(
        counts, np.array([[0.0], [-1.0], [-0.5]]), lad)
    assert spectral_trust_verdict(bad).startswith("NOT TRUSTWORTHY")

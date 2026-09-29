"""Gates for ``spectral_shell``, the pooled denominator-shell estimator.

The model: band A adds ``a_i · Σ_k w_k (E_Ak − E_i + Ω)^(−β)`` to state i, one
(β, Ω) pooled over the requested states, one amplitude per state
(``gw.band_extrapolation``, POOLED DENOMINATOR SHELL).  These tests pin:
the fit recovers a model it was built from, the compressed shell sums equal
the raw sums, the weights keep Σ Hermitian and affine, the no-domain rules,
the refusals, the payload and the cost.

The measured accuracy (BANDEX Si 4³ against the complete basis) is not
re-checked here: it is reproduced from the study's stored samples by
``sandbox:runs/DEV/602_bandx2_20260928/analysis/rescore.py``.
"""
from __future__ import annotations

import inspect
import time

import numpy as np
import pytest

from gw.band_extrapolation import (
    BAND_EXTRAPOLATION_ESTIMATORS,
    BAND_EXTRAPOLATION_ESTIMATOR_DEFAULT,
    BRACKET_FRACTIONS,
    SHELL_BETA_GRID,
    SHELL_FAIL_NO_FIT,
    SHELL_FAIL_POLE,
    SHELL_OK,
    SHELL_OMEGA_GRID_EV,
    SPECTRAL_EXTRAP_DATASETS,
    BandExtrapolationRefused,
    build_band_ladder,
    fit_band_extrapolation_spectral,
    format_spectral_report,
    plan_band_brackets,
    spectral_h5_payload,
    spectral_trust_verdict,
)
from common.units import RYD_TO_EV


def _ladder(nk=6, nb=80, n_target=2000, seed=0):
    """A free-electron-like DFT ladder with k dispersion, bands sorted per k."""
    rng = np.random.default_rng(seed)
    n = np.arange(1, nb + 1, dtype=np.float64)
    e = (-12.0 + 3.0 * n ** (2.0 / 3.0))[None, :] + rng.normal(0, 0.6, (nk, nb))
    e = np.sort(e, axis=1)
    return build_band_ladder(enk_ry=e / RYD_TO_EV, kweights=None,
                             n_target=n_target)


def _raw_shell(lad, lo, hi, e_i, beta, omega):
    """Σ over every raw (band, k) term of absolute bands (lo, hi]."""
    e, w = lad._terms(lo, hi)
    x = (e[None, :] - np.asarray(e_i)[:, None] + omega) / lad.estar_ev
    return (w[None, :] * x ** -beta).sum(axis=1)


def _model_samples(lad, counts, e_i, beta, omega, s_inf, amp):
    """S_i(N) = S_inf,i − a_i · G_i(N, N_T): the estimator's own model."""
    return np.stack([s_inf - amp * _raw_shell(lad, c, lad.n_target, e_i,
                                              beta, omega) for c in counts])


def test_one_estimator_name():
    assert BAND_EXTRAPOLATION_ESTIMATORS == ("spectral_shell",)
    assert BAND_EXTRAPOLATION_ESTIMATOR_DEFAULT == "spectral_shell"


def test_default_fractions_are_the_owner_ruling():
    assert BRACKET_FRACTIONS == (0.70, 0.85)


@pytest.mark.parametrize("beta,omega", [(3.0, 10.0), (4.5, 24.0), (2.25, 0.0)])
def test_recovers_the_model_it_was_built_from(beta, omega):
    """Data from the model on a grid point: the fit returns that point and S_inf."""
    lad = _ladder()
    counts = (56, 68, 80)
    rng = np.random.default_rng(1)
    e_i = rng.uniform(-10.0, 8.0, 40)
    s_inf = rng.normal(-2.0, 0.5, 40)
    amp = rng.uniform(0.5, 3.0, 40)
    S = _model_samples(lad, counts, e_i, beta, omega, s_inf, amp)
    fit = fit_band_extrapolation_spectral(counts, S, lad, e_state_ev=e_i)
    assert (fit.beta, fit.omega_ev) == (beta, omega)
    assert fit.residual_ev < 1e-12
    assert np.max(np.abs(fit.s_inf - s_inf)) < 1e-10
    assert fit.n_failed == 0


def test_compressed_shell_sums_equal_the_raw_sums():
    """The composite Gauss compression is exact to roundoff, near a pole too."""
    lad = _ladder(nb=60, n_target=40000)
    floor = lad.floor_ev(40)
    e_i = np.array([floor - 1e-3, floor - 0.3, floor - 4.0, floor - 25.0])
    for beta in (SHELL_BETA_GRID[0], 3.3, SHELL_BETA_GRID[1]):
        for omega in (0.0, 7.0, SHELL_OMEGA_GRID_EV[1]):
            e_ref = float(np.max(e_i - omega))
            for lo, hi in ((40, 50), (40, 60), (60, lad.n_target)):
                de, w = lad.shell_rule(lo, hi, e_ref)
                x = (de[None, :] + (e_ref - e_i + omega)[:, None]) / lad.estar_ev
                got = (w[None, :] * x ** -beta).sum(axis=1)
                want = _raw_shell(lad, lo, hi, e_i, beta, omega)
                assert np.max(np.abs(got / want - 1.0)) < 1e-12, (lo, hi, beta, omega)


def test_weights_are_real_affine_and_reproduce_s_inf():
    lad = _ladder()
    counts = (56, 68, 80)
    rng = np.random.default_rng(2)
    e_i = rng.uniform(-10.0, 8.0, (3, 5))
    S = (_model_samples(lad, counts, e_i.ravel(), 3.5, 12.0,
                        rng.normal(size=15), rng.uniform(1, 2, 15))
         .reshape(3, 3, 5) + 1j * rng.normal(0, 0.01, (3, 3, 5)))
    fit = fit_band_extrapolation_spectral(counts, S, lad, e_state_ev=e_i)
    c = fit.weights()
    assert np.isrealobj(c) and c.shape == (3, 3, 5)
    assert np.max(np.abs(c.sum(axis=0) - 1.0)) < 1e-14
    assert np.all(c[1] == 0.0), "N2 fixes the shape and carries no weight"
    assert np.max(np.abs((c * S).sum(axis=0) - fit.s_inf)) < 1e-12
    # The symmetrised off-diagonal rule r_ij = (r_i + r_j)/2 keeps Σ Hermitian.
    r = fit.tail_ratio.reshape(-1)[:4]
    A = rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4))
    A = A + np.conj(np.swapaxes(A, 1, 2))
    rij = 0.5 * (r[:, None] + r[None, :])
    out = A[2] + (A[2] - A[0]) * rij
    assert np.max(np.abs(out - out.conj().T)) == 0.0


def test_a_constant_offset_moves_s_inf_and_nothing_else():
    lad = _ladder()
    counts = (56, 68, 80)
    e_i = np.linspace(-9.0, 6.0, 12)
    S = _model_samples(lad, counts, e_i, 3.0, 8.0, -1.0, 2.0)
    f1 = fit_band_extrapolation_spectral(counts, S, lad, e_state_ev=e_i)
    f2 = fit_band_extrapolation_spectral(counts, S + 0.37, lad, e_state_ev=e_i)
    assert (f1.beta, f1.omega_ev) == (f2.beta, f2.omega_ev)
    assert np.max(np.abs(f2.s_inf - f1.s_inf - 0.37)) < 1e-12


def test_a_state_above_the_extrapolated_bands_keeps_its_sum():
    lad = _ladder()
    counts = (56, 68, 80)
    floor = lad.floor_ev(56)
    # Above the floor by more than the largest Omega: a pole at every Omega.
    e_i = np.array([-5.0, 0.0, 5.0, floor + SHELL_OMEGA_GRID_EV[1] + 1.0])
    S = _model_samples(lad, counts, e_i[:3], 3.0, 8.0, -1.0, 2.0)
    S = np.concatenate([S, np.array([[-0.9], [-0.8], [-0.7]])], axis=1)
    fit = fit_band_extrapolation_spectral(counts, S, lad, e_state_ev=e_i)
    assert list(fit.failure) == [SHELL_OK] * 3 + [SHELL_FAIL_POLE]
    assert not fit.fit_mask[3], "a state outside the domain is not pooled"
    assert fit.s_inf[3] == S[2, 3] and fit.tail_ratio[3] == 0.0
    assert "NO TAIL" in fit.failure_report()
    assert "1 of 4" in fit.failure_report()


def test_no_state_in_the_domain_means_no_fit_and_no_tail():
    lad = _ladder()
    counts = (56, 68, 80)
    e_i = np.full(3, lad.floor_ev(56) + 1.0)
    S = np.stack([np.full(3, v) for v in (-1.0, -0.9, -0.85)])
    fit = fit_band_extrapolation_spectral(counts, S, lad, e_state_ev=e_i)
    assert np.isnan(fit.beta) and np.isnan(fit.omega_ev)
    assert np.all(fit.failure == SHELL_FAIL_NO_FIT)
    assert np.array_equal(fit.s_inf, S[2])
    assert spectral_trust_verdict(fit).startswith("NOT TRUSTWORTHY")


def test_fit_mask_restricts_the_pool_only():
    """The pool fixes (β, Ω); every state in the domain still gets its tail."""
    lad = _ladder()
    counts = (56, 68, 80)
    e_i = np.linspace(-9.0, 6.0, 10)
    S = _model_samples(lad, counts, e_i, 4.0, 16.0, -1.0, 1.5)
    mask = np.zeros(10, dtype=bool)
    mask[:5] = True
    fit = fit_band_extrapolation_spectral(counts, S, lad, e_state_ev=e_i,
                                          fit_mask=mask)
    assert (fit.beta, fit.omega_ev) == (4.0, 16.0)
    assert int(fit.fit_mask.sum()) == 5 and fit.n_failed == 0
    assert np.all(fit.tail_ratio > 0.0)


def test_refuses_when_n3_reaches_the_basis():
    lad = _ladder(nb=80, n_target=80)
    S = np.zeros((3, 2))
    with pytest.raises(BandExtrapolationRefused, match="finite-basis endpoint"):
        fit_band_extrapolation_spectral((56, 68, 80), S, lad,
                                        e_state_ev=np.zeros(2))


@pytest.mark.parametrize("counts", [(56, 68), (56, 56, 80), (68, 56, 80)])
def test_refuses_bad_counts(counts):
    lad = _ladder()
    with pytest.raises(ValueError):
        fit_band_extrapolation_spectral(counts, np.zeros((len(counts), 2)), lad,
                                        e_state_ev=np.zeros(2))


def test_the_state_energy_is_a_required_input():
    """The model's denominator reads E_i; a call without it must not run."""
    params = inspect.signature(fit_band_extrapolation_spectral).parameters
    assert params["e_state_ev"].default is inspect.Parameter.empty
    assert params["e_state_ev"].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_ladder_is_built_from_dft_only():
    """build_band_ladder has no input through which Σ could reach the ladder."""
    params = set(inspect.signature(build_band_ladder).parameters)
    assert params == {"enk_ry", "kweights", "n_target", "b0", "fit_window",
                      "estar_window"}


def test_fit_is_fast_on_a_large_deck():
    """Owner 2026-09-28: the extrapolation must not take seconds.

    1200 states, 400 DFT bands at 12 k, a 152012-band Weyl tail (the CrI3
    16x16 N_T): fit + apply well under a second.
    """
    lad = _ladder(nk=12, nb=400, n_target=152012, seed=3)
    counts = (280, 340, 400)
    rng = np.random.default_rng(4)
    e_i = rng.uniform(-10.0, 10.0, 1200)
    S = np.stack([-1.0 + 0.3 / c + 1e-3 * rng.normal(size=1200)
                  for c in counts]).astype(complex)
    t0 = time.perf_counter()
    fit = fit_band_extrapolation_spectral(counts, S, lad, e_state_ev=e_i)
    wall = time.perf_counter() - t0
    assert np.isfinite(fit.beta)
    assert wall < 1.0, f"fit + apply took {wall:.2f} s"


def test_payload_and_report():
    lad = _ladder()
    e = np.tile(np.linspace(1.0, 10.0, 80), (4, 1)) / RYD_TO_EV
    plan = plan_band_brackets(enabled=True, enk_ry=e, n_occ=8, nb_logical=80,
                              nb_padded=80)
    assert plan.counts == (56, 68, 80)
    e_i = np.linspace(-9.0, 6.0, 8).reshape(2, 4)
    S = _model_samples(lad, plan.counts, e_i.ravel(), 3.0, 8.0, -1.0,
                       2.0).reshape(3, 2, 4)
    fit = fit_band_extrapolation_spectral(plan.counts, S, lad, e_state_ev=e_i)
    pay = spectral_h5_payload(plan, fit)
    assert set(pay["arrays"]) == set(SPECTRAL_EXTRAP_DATASETS)
    assert pay["attrs"]["pooled_beta"] == 3.0
    assert pay["attrs"]["pooled_omega_ev"] == 8.0
    assert np.allclose(pay["arrays"]["sigma_c_extrap_beta_kn"].real, 3.0)
    text = format_spectral_report(plan, fit, states=[("VBM", (0, 1))])
    assert "pooled beta = 3.00" in text and "Omega = 8.0 eV" in text
    assert "[VBM]" in text

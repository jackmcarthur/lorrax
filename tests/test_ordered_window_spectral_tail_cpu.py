"""Dense small-band oracle of live one-sided TD shell-tail algebra, CPU only."""
from dataclasses import replace

import numpy as np
import pytest

from gw.band_extrapolation import (BandLadder, BandExtrapolationRefused,
    fit_band_extrapolation_spectral, spectral_tail_ratio)
from gw.ppm_sigma import ordered_window_spectral_tail, _ordered_tail_brackets


def ladder():
    energies = np.arange(1., 21.)[:, None]+np.array([[0., .2]])
    return BandLadder(e_dft_ev=energies, e_weyl_ev=np.arange(21., 41.),
        w_k=np.array([.3, .7]), e0_ev=0., n0=0., c_ev=1., r2=1.,
        estar_ev=10., n_dft=20, n_target=40, fit_window=(10, 20))


def moment(lad, lo, hi, e, omega=2.):
    nodes, weights = lad._terms(lo, hi)
    return np.sum(weights/(nodes-np.asarray(e)[..., None]+omega)**3, axis=-1)


def reference(e):
    lad, counts = ladder(), [6, 9, 12]
    first = np.ones(np.shape(e), complex)*(.2+.4j)
    amplitude = np.arange(np.size(e)).reshape(np.shape(e))+.5+.2j
    sums = np.stack([first, first+amplitude*moment(lad, 6, 9, e),
                    first+amplitude*moment(lad, 6, 12, e)])
    fit = fit_band_extrapolation_spectral(counts, sums, lad, e_state_ev=e)
    assert fit.omega_ev == 2.
    return dict(fit=fit, input_frame='unit-oracle-only', ladder_shift_ev=0.)


def test_diagonal_recovers_native_complex_estimator():
    e = np.array([[-2., -1., 0.], [-1.5, -.5, .5]])
    ref = reference(e)
    fit = ref['fit']
    pairs = np.zeros((3, 2, 2, 3, 3), complex)
    for n in range(3):
        for slot in range(2):
            for k in range(2):
                np.fill_diagonal(pairs[n, slot, k], fit.s_at_counts[n, k])
    out, _ = ordered_window_spectral_tail(pairs[0], pairs[2], e, ref)
    np.testing.assert_allclose(np.diagonal(out, axis1=-2, axis2=-1),
        np.broadcast_to(fit.s_inf, (2, *e.shape)), rtol=1.e-14, atol=1.e-15)


def test_complex_offdiagonal_dense_shell_oracle_not_arithmetic_average():
    e = np.array([[-2., -1., 0.], [-1.5, -.5, .5]])
    ref, rng = reference(e), np.random.default_rng(9)
    first = rng.normal(size=(2, 2, 3, 3))+1j*rng.normal(size=(2, 2, 3, 3))
    last = first+rng.normal(size=first.shape)+1j*rng.normal(size=first.shape)
    out, receipt = ordered_window_spectral_tail(first, last, e, ref)
    ratio = moment(ref['fit'].ladder, 12, 40, e)/moment(ref['fit'].ladder, 6, 12, e)
    expected = last.copy()
    for slot in range(2):
        for k in range(2):
            for a in range(3):
                for b in range(3):
                    expected[slot, k, a, b] += (last-first)[slot, k, a, b]*ratio[k, a if slot == 0 else b]
    np.testing.assert_allclose(out, expected, rtol=1.e-14, atol=1.e-14)
    mean = last+(last-first)*(ratio[:, :, None]+ratio[:, None, :])[None]/2
    assert np.max(abs(out-mean)) > .001
    assert receipt['destination'] == 'full_virtual_hermitian_material_self_energy_only'


def test_current_energy_and_uniform_ladder_shift_not_frozen_band_ratio():
    e = np.array([[-2., -1., 0.]])
    ref = reference(e)
    first, last = np.zeros((2, 1, 3, 3), complex), np.ones((2, 1, 3, 3), complex)
    old, _ = ordered_window_spectral_tail(first, last, e, ref)
    live, _ = ordered_window_spectral_tail(first, last, e+.5, ref)
    assert np.max(abs(old-live)) > 1.e-4
    shifted, _ = ordered_window_spectral_tail(first, last, e+3., dict(ref, ladder_shift_ev=3.))
    np.testing.assert_allclose(shifted, old, rtol=1.e-14)


def test_degenerate_subspace_covariance_of_both_shell_slots():
    e = np.array([[-1., -1., -1.]])
    ref, rng = reference(e), np.random.default_rng(12)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3)))
    first = rng.normal(size=(2, 1, 3, 3))+1j*rng.normal(size=(2, 1, 3, 3))
    last = rng.normal(size=first.shape)+1j*rng.normal(size=first.shape)
    out, _ = ordered_window_spectral_tail(first, last, e, ref)
    rotate = lambda a: q.conj().T @ a @ q
    rotated, _ = ordered_window_spectral_tail(rotate(first), rotate(last), e, ref)
    np.testing.assert_allclose(rotated, rotate(out), rtol=1.e-14, atol=1.e-14)


def test_crossed_shell_is_refused_not_clipped():
    with pytest.raises(BandExtrapolationRefused, match='domain'):
        spectral_tail_ratio(ladder(), [6, 9, 12], [12.], beta=3., omega_ev=2.)


def test_tail_bracket_order_preserves_active_and_native_counts_with_pad():
    ref = dict(fit=replace(reference(np.array([[-1.]]))['fit'], counts=np.array([525, 638, 750])))
    brackets, labels, matrices, first = _ordered_tail_brackets(ref,
        b_lo=128, hi=132, n_insert=16, nb=752)
    assert brackets == ((128, 136), (0, 128), (148, 541), (541, 654), (654, 768))
    assert labels == (4, 132, 525, 638, 750)
    assert matrices == (0, 2, 4) and first == 2
    # Only the inserted pad136:148 is skipped. All other nonzero original
    # and identity panels are covered exactly once.
    covered = np.zeros(768, int)
    for lo, hi in brackets:
        covered[lo:hi] += 1
    assert np.all(covered[:136] == 1) and np.all(covered[148:] == 1)
    assert np.all(covered[136:148] == 0)


def test_real_transition_window_cannot_enter_virtual_tail_shell():
    with pytest.raises(ValueError, match='above active'):
        _ordered_tail_brackets(reference(np.array([[-1.]])),
            b_lo=3, hi=7, n_insert=4, nb=20)

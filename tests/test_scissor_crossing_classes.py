"""Fermi-crossing bands belong to NEITHER scissor fit class.

The rule under test (owner ruling, 2026-08-16), for the metallic QSGW
scissor:

    valence class    = bands BELOW the LOWEST band that crosses E_F
    conduction class = bands ABOVE the HIGHEST band that crosses E_F
    crossing bands   = in neither fit class

Three cells, in the order the brief names them:

1. **The discriminating metallic cell.**  A synthetic semicore + crossing
   pair + conduction spectrum whose valence QP correction is exactly affine.
   With the crossing pair IN the valence class the fit is dragged off that
   line and mispredicts the deep semicore by eV; with it excluded the fit
   recovers the law to float64 and the semicore prediction is exact.  The
   sanity scale is claim 0212, sodium ``02_soc48b_qsgw_mpa``: the
   ``[-5,+5]`` val fit was 100% crossing samples (n_v = 1024, alpha =
   0.9100) and predicted the 2s semicore **+4.84 eV** where BerkeleyGW's
   Eqp0 is **-12.86 eV** — wrong by 17.5 eV and wrong in sign.

2. **Insulating byte-compat.**  With step occupations nothing crosses, and
   the three-way rule collapses onto exactly the ``arange(nb) < nelec``
   index mask the driver freezes.  Compared with ``==`` on every
   ``ScissorFit`` field, not ``allclose``: the insulating path is the
   production path for every non-metal deck and must not move by one ulp.

3. **The empty-class law survives (commit ``bf57701b``).**  Sodium's
   ``[-5,+5]`` window has NO true-valence band in range once the crossing
   pair is excluded, so the valence class is empty and its law is the
   identity.  That is the point of the change, not a regression: an
   identity refuses to extrapolate, where the old crossing-anchored line
   extrapolated confidently and wrongly.

Plus the classifier's own conventions: MP1 overshoot (``f`` is never
clipped, so occupied bands carry ``f > 1``), the mixed full/empty band, and
the refusal on a band axis that is not energy-sorted.

``src/gw/scissor.py`` is loaded FROM ITS PATH for the same isolation reason
``test_scissor_weights.py`` gives: this module needs only numpy and the test
does not need the rest of the GW package.
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_SCISSOR = (pathlib.Path(__file__).resolve().parents[1]
            / "src" / "gw" / "scissor.py")
_spec = importlib.util.spec_from_file_location(
    "_lorrax_scissor_classes", _SCISSOR)
scissor = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = scissor
_spec.loader.exec_module(scissor)

classify_scissor_bands = scissor.classify_scissor_bands
fit_scissor = scissor.fit_scissor
full_bz_k_weights = scissor.full_bz_k_weights
ScissorBandClasses = scissor.ScissorBandClasses
qsgw_out_of_range_energies = scissor.qsgw_out_of_range_energies


# ---------------------------------------------------------------------------
# A sodium-shaped synthetic deck
# ---------------------------------------------------------------------------
#
# Bands, energies relative to mu, chosen to sit where sodium's really do
# (claim 0212: 2s at E_DFT - mu = -52.16 eV, the 2p one window deeper than
# [-5,+5] and inside [-28,+7], the Fermi pair straddling zero):
#
#   0-1    2s semicore     ~ -52 eV      OUT of the [-28,+26] window
#   2-7    2p semicore     ~ -25 eV      in
#   8-9    Fermi crossing  ~   0 eV      in, protected, FRACTIONAL f
#   10-15  conduction      ~ +5..+25 eV  in
#
_N_K = 8
_I_2S = (0, 2)
_I_2P = (2, 8)
_I_CROSS = (8, 10)
_I_COND = (10, 16)
_NB = 16

# The truth the valence class is supposed to recover.  alpha = 1.2 / beta =
# -0.45 eV is deliberately the neighbourhood of claim 0212's repaired val
# fit (alpha 0.9100 -> 1.1978, beta -0.0015 -> -0.4565 eV).
_ALPHA_V, _BETA_V = 1.20, -0.45
_ALPHA_C, _BETA_C = 1.05, -1.30
# What the Fermi pair actually does: nothing like either line.  +1.88 eV is
# the measured mean QP correction on sodium's converged protected pair
# (metallic-mpa-screening.md 7.4).
_DELTA_CROSS = 1.88


def _na_like_deck():
    """Return ``(E_dft, E_qp, f_kn)`` for the synthetic deck above."""
    rng = np.random.default_rng(20260816)
    E = np.empty((_N_K, _NB), dtype=np.float64)

    def fill(lo, hi, e0, spread):
        n = hi - lo
        base = np.linspace(e0 - spread, e0 + spread, n)
        E[:, lo:hi] = base[None, :] + rng.uniform(
            -0.15, 0.15, size=(_N_K, n))

    fill(*_I_2S, -52.16, 0.30)
    fill(*_I_2P, -25.00, 2.00)
    fill(*_I_CROSS, 0.00, 1.20)
    fill(*_I_COND, 15.00, 10.00)
    # Bands must be ascending in n at each k -- the classifier says so, and
    # so does every producer (eigvalsh, QE).
    E = np.sort(E, axis=1)

    dE = np.empty_like(E)
    v = slice(0, _I_CROSS[0])
    c = slice(_I_CROSS[1], _NB)
    dE[:, v] = (_ALPHA_V - 1.0) * E[:, v] + _BETA_V
    dE[:, c] = (_ALPHA_C - 1.0) * E[:, c] + _BETA_C
    dE[:, slice(*_I_CROSS)] = _DELTA_CROSS

    # MP1 occupations, unclipped, from a tanh-ish saturating profile: deep
    # bands saturate to exactly 1.0, high bands to exactly 0.0, and the two
    # crossing bands carry genuinely fractional cells.
    f = np.zeros_like(E)
    f[:, :_I_CROSS[0]] = 1.0
    f[:, slice(*_I_CROSS)] = np.clip(0.5 - 0.35 * E[:, slice(*_I_CROSS)],
                                     0.02, 0.98)
    return E, E + dE, f


def _index_mask(nk, nb, nelec):
    """The frozen driver mask: ``arange(nb) < meta.nelec``, broadcast."""
    return np.broadcast_to((np.arange(nb) < int(nelec))[None, :], (nk, nb))


def _band_window(nb, lo, hi):
    """``(nb,)`` in-range mask for bands ``[lo, hi)``."""
    idx = np.arange(nb)
    return (idx >= lo) & (idx < hi)


# ---------------------------------------------------------------------------
# 1.  The discriminating metallic cell
# ---------------------------------------------------------------------------

def test_the_crossing_pair_is_classified_out_of_both_fit_classes():
    _, _, f = _na_like_deck()
    cls = classify_scissor_bands(f)
    assert (cls.valence_stop, cls.conduction_start) == _I_CROSS
    assert cls.n_crossing == 2
    val_kn, cross_kn = cls.masks(f.shape)
    assert val_kn[:, :_I_CROSS[0]].all()
    assert not val_kn[:, _I_CROSS[0]:].any()
    assert cross_kn[:, slice(*_I_CROSS)].all()
    assert cross_kn.sum() == _N_K * 2


def test_excluding_the_crossing_pair_repairs_the_semicore_extrapolation():
    """The whole point, in one cell.

    Window ``[-28, +26]`` about mu: the 2p, the Fermi pair and the whole
    conduction manifold are in range; the 2s is not and is what the scissor
    must predict.  The valence QP correction IS affine here by construction,
    so a fit over true-valence samples alone must recover it to float64 —
    and any fit that also swallows the Fermi pair cannot.
    """
    E_dft, E_qp, f = _na_like_deck()
    cls = classify_scissor_bands(f)
    in_range = np.broadcast_to(
        _band_window(_NB, _I_2P[0], _NB)[None, :], E_dft.shape)
    w = full_bz_k_weights(_N_K)

    # OLD: "occupied at DFT occupation" index cut.  nelec = 10 puts BOTH
    # crossing bands in the valence class -- exactly sodium's nval = 10.
    old_val = _index_mask(_N_K, _NB, _I_CROSS[1])
    old = fit_scissor(E_dft, E_qp, valence_mask_kn=old_val,
                      fit_mask_kn=in_range, k_weights=w)

    # NEW: three-way.  Crossing bands enter neither class.
    new_val, new_cross = cls.masks(E_dft.shape)
    new = fit_scissor(E_dft, E_qp, valence_mask_kn=new_val,
                      fit_mask_kn=in_range & ~new_cross, k_weights=w)

    # Sample bookkeeping: the crossing pair is the whole difference.
    assert old.n_fit_v == _N_K * (_I_2P[1] - _I_2P[0] + 2)
    assert new.n_fit_v == _N_K * (_I_2P[1] - _I_2P[0])
    assert old.n_fit_v - new.n_fit_v == _N_K * 2
    # ...and the conduction class is untouched in THIS window, which is why
    # the valence side is the discriminator here.
    assert old.n_fit_c == new.n_fit_c == _N_K * (_NB - _I_COND[0])

    # The new valence law IS the truth.
    assert new.alpha_v == pytest.approx(_ALPHA_V, abs=1e-12)
    assert new.beta_v_ev == pytest.approx(_BETA_V, abs=1e-12)
    assert new.rmse_v_ev < 1e-12
    # The old one is not, and its residual says so out loud: 0.20 eV of
    # unexplained valence scatter against the new fit's float64 zero.  (The
    # real 18_ arm's crossing-contaminated val fit printed rmse = 0.209 eV
    # on the same shaped problem.)
    assert abs(old.alpha_v - _ALPHA_V) > 0.05
    assert old.rmse_v_ev > 0.1
    assert old.rmse_v_ev > 1.0e6 * max(new.rmse_v_ev, 1e-300)

    # Prediction on the out-of-window 2s semicore.
    truth = (_ALPHA_V - 1.0) * E_dft + _BETA_V
    d_new = new.predict(E_dft, new_val, crossing_mask=new_cross)
    d_old = old.predict(E_dft, old_val)
    semicore = slice(*_I_2S)
    err_new = float(np.abs(d_new[:, semicore] - truth[:, semicore]).max())
    err_old = float(np.abs(d_old[:, semicore] - truth[:, semicore]).max())
    assert err_new < 1e-11
    assert err_old > 2.0
    # Stated as the ratio the claim reports, so a regression that halves
    # the damage still fails.
    assert err_old > 1.0e9 * max(err_new, 1e-300)


def test_the_crossing_bands_get_no_extrapolation_at_all():
    """A band we refused to FIT is a band we refuse to EXTRAPOLATE."""
    E_dft, E_qp, f = _na_like_deck()
    cls = classify_scissor_bands(f)
    in_range = np.broadcast_to(
        _band_window(_NB, _I_2P[0], _NB)[None, :], E_dft.shape)
    val_kn, cross_kn = cls.masks(E_dft.shape)
    fit = fit_scissor(E_dft, E_qp, valence_mask_kn=val_kn,
                      fit_mask_kn=in_range & ~cross_kn,
                      k_weights=full_bz_k_weights(_N_K))
    delta = fit.predict(E_dft, val_kn, crossing_mask=cross_kn)
    assert np.array_equal(delta[:, slice(*_I_CROSS)],
                          np.zeros((_N_K, 2)))
    # ...and nothing else was zeroed.
    assert np.all(delta[:, :_I_CROSS[0]] != 0.0)
    assert np.all(delta[:, _I_CROSS[1]:] != 0.0)


# ---------------------------------------------------------------------------
# 2.  Insulating byte-compat
# ---------------------------------------------------------------------------

def _insulating_deck(nk=6, nb=12, nocc=5):
    rng = np.random.default_rng(4242)
    E = np.sort(rng.uniform(-20.0, -2.0, size=(nk, nocc)), axis=1)
    Ec = np.sort(rng.uniform(2.0, 30.0, size=(nk, nb - nocc)), axis=1)
    E = np.concatenate([E, Ec], axis=1)
    dE = np.where(np.arange(nb)[None, :] < nocc,
                  -0.3 * E - 1.1, 0.2 * E + 0.7)
    f = (np.arange(nb)[None, :] < nocc).astype(np.float64)
    f = np.broadcast_to(f, (nk, nb)).copy()
    return E, E + dE, f, nocc


def test_step_occupations_reproduce_the_index_mask_exactly():
    E, _, f, nocc = _insulating_deck()
    cls = classify_scissor_bands(f)
    assert cls.valence_stop == cls.conduction_start == nocc
    assert cls.n_crossing == 0
    val_kn, cross_kn = cls.masks(E.shape)
    assert np.array_equal(val_kn, _index_mask(*E.shape, nocc))
    assert not cross_kn.any()


def test_insulating_fit_is_bitwise_identical_under_the_new_masks():
    """Field-by-field ``==``.  Not ``allclose``: this is the default path."""
    E_dft, E_qp, f, nocc = _insulating_deck()
    nk, nb = E_dft.shape
    in_range = np.broadcast_to(
        _band_window(nb, 1, nb - 2)[None, :], E_dft.shape)
    w = full_bz_k_weights(nk)

    old_val = _index_mask(nk, nb, nocc)
    old = fit_scissor(E_dft, E_qp, valence_mask_kn=old_val,
                      fit_mask_kn=in_range, k_weights=w)

    cls = classify_scissor_bands(f)
    new_val, new_cross = cls.masks(E_dft.shape)
    new = fit_scissor(E_dft, E_qp, valence_mask_kn=new_val,
                      fit_mask_kn=in_range & ~new_cross, k_weights=w)

    assert new == old                       # frozen dataclass, field-wise ==
    assert new.alpha_v == old.alpha_v       # spelled out so a failure names
    assert new.beta_v_ev == old.beta_v_ev   # the field that moved
    assert new.alpha_c == old.alpha_c
    assert new.beta_c_ev == old.beta_c_ev
    assert new.n_fit_v == old.n_fit_v
    assert new.n_fit_c == old.n_fit_c

    d_old = old.predict(E_dft, old_val)
    d_new = new.predict(E_dft, new_val, crossing_mask=new_cross)
    assert np.array_equal(d_new, d_old)


def test_predict_without_a_crossing_mask_is_the_historical_expression():
    """``crossing_mask=None`` must not perturb the two-way arithmetic."""
    E_dft, E_qp, f, nocc = _insulating_deck()
    nk, nb = E_dft.shape
    vm = _index_mask(nk, nb, nocc)
    fit = fit_scissor(E_dft, E_qp, valence_mask_kn=vm,
                      fit_mask_kn=np.ones_like(vm),
                      k_weights=full_bz_k_weights(nk))
    a = fit.predict(E_dft, vm)
    b = fit.predict(E_dft, vm, crossing_mask=None)
    c = fit.predict(E_dft, vm, crossing_mask=np.zeros_like(vm))
    assert np.array_equal(a, b)
    assert np.array_equal(a, c)


# ---------------------------------------------------------------------------
# 3.  The empty-class law is preserved (bf57701b)
# ---------------------------------------------------------------------------

def test_a_window_holding_only_crossing_bands_gives_the_identity():
    """Sodium's ``[-5,+5]`` arm, which is where this rule bites hardest.

    The only in-range bands ARE the Fermi pair.  Excluding them empties BOTH
    classes, so both laws are the identity and every scissored band keeps
    E_DFT.  That is the honest answer: the old fit here had n_v = 1024
    samples, all of them crossing, and its 2s extrapolation was wrong by
    17.5 eV in the wrong direction (claim 0212).
    """
    E_dft, E_qp, f = _na_like_deck()
    cls = classify_scissor_bands(f)
    in_range = np.broadcast_to(
        _band_window(_NB, *_I_CROSS)[None, :], E_dft.shape)
    val_kn, cross_kn = cls.masks(E_dft.shape)
    w = full_bz_k_weights(_N_K)

    new = fit_scissor(E_dft, E_qp, valence_mask_kn=val_kn,
                      fit_mask_kn=in_range & ~cross_kn, k_weights=w)
    assert (new.n_fit_v, new.n_fit_c) == (0, 0)
    assert (new.alpha_v, new.beta_v_ev) == (1.0, 0.0)
    assert (new.alpha_c, new.beta_c_ev) == (1.0, 0.0)
    delta = new.predict(E_dft, val_kn, crossing_mask=cross_kn)
    assert np.array_equal(delta, np.zeros_like(delta))

    # The contrast, on the same window: the old mask fits the crossing pair
    # as valence and hands the 2s semicore a confident wrong number.
    old_val = _index_mask(_N_K, _NB, _I_CROSS[1])
    old = fit_scissor(E_dft, E_qp, valence_mask_kn=old_val,
                      fit_mask_kn=in_range, k_weights=w)
    assert old.n_fit_v == _N_K * 2
    truth = (_ALPHA_V - 1.0) * E_dft + _BETA_V
    d_old = old.predict(E_dft, old_val)
    semicore = slice(*_I_2S)
    assert float(np.abs(
        d_old[:, semicore] - truth[:, semicore]).max()) > 5.0


def test_a_single_true_valence_sample_is_still_a_rigid_shift():
    """The other half of bf57701b's law survives the new masks."""
    E_dft, E_qp, f = _na_like_deck()
    cls = classify_scissor_bands(f)
    one_band = np.broadcast_to(
        _band_window(_NB, _I_2P[1] - 1, _NB)[None, :], E_dft.shape)
    val_kn, cross_kn = cls.masks(E_dft.shape)
    # One k only, so the valence class has exactly one sample.
    fm = (one_band & ~cross_kn).copy()
    fm[1:, :] = False
    fit = fit_scissor(E_dft, E_qp, valence_mask_kn=val_kn, fit_mask_kn=fm,
                      k_weights=full_bz_k_weights(_N_K))
    assert fit.n_fit_v == 1
    assert fit.alpha_v == 1.0
    assert fit.beta_v_ev == pytest.approx(
        float((E_qp - E_dft)[0, _I_2P[1] - 1]), abs=1e-12)


# ---------------------------------------------------------------------------
# 4.  Classifier conventions
# ---------------------------------------------------------------------------

def test_mp1_overshoot_counts_as_saturated_not_as_fractional():
    """``f_kn`` is never clipped, so occupied bands carry ``f > 1``.

    A test spelled ``0 < f < 1`` would call a 1.008 cell "not fractional"
    for the wrong reason; one spelled ``f == 1.0`` would call a
    0.9999999999 cell fractional and turn a whole semicore shell into a
    crossing band.  Both are covered here.
    """
    f = np.array([
        [1.0251, 1.0080, 0.6, -0.0080, -0.0031],
        [1.0000, 1.0000, 0.4,  0.0000,  0.0000],
        [0.9999999999, 1.0002, 0.5, 1.0e-12, 0.0],
    ])
    cls = classify_scissor_bands(f)
    assert (cls.valence_stop, cls.conduction_start) == (2, 3)
    assert cls.n_crossing == 1


def test_a_band_full_at_one_k_and_empty_at_another_is_a_crossing_band():
    """No cell is fractional, yet the band manifestly crosses E_F.

    This is why the classifier tests SATURATION per cell and then requires
    every cell of a class to agree, rather than testing ``min f`` and
    ``max f`` against the open interval (0, 1) — under MP1 overshoot both of
    those lie outside (0, 1) for a band that runs from full to empty.
    """
    f = np.array([
        [1.004, 1.004, 0.0],
        [1.004, -0.004, 0.0],
    ])
    cls = classify_scissor_bands(f)
    assert (cls.valence_stop, cls.conduction_start) == (1, 2)


def test_an_unsorted_band_axis_is_refused_not_guessed():
    f = np.array([[0.5, 1.0, 0.5], [0.5, 1.0, 0.5]])
    with pytest.raises(ValueError, match="energy-sorted"):
        classify_scissor_bands(f)


def test_every_band_crossing_leaves_both_classes_empty():
    f = np.full((3, 4), 0.5)
    cls = classify_scissor_bands(f)
    assert (cls.valence_stop, cls.conduction_start) == (0, 4)
    val_kn, cross_kn = cls.masks(f.shape)
    assert not val_kn.any()
    assert cross_kn.all()


def test_classes_are_width_agnostic_so_a_padded_occupation_table_is_safe():
    """The occupation table is the PADDED parallel-transport manifold; the
    fit sees the active window.  Boundary indices carry across both."""
    f = np.zeros((2, 20))
    f[:, :6] = 1.0
    f[:, 6:8] = 0.5
    cls = classify_scissor_bands(f)
    assert (cls.valence_stop, cls.conduction_start) == (6, 8)
    for nb in (8, 12, 48):
        val_kn, cross_kn = cls.masks((5, nb))
        assert val_kn.shape == cross_kn.shape == (5, nb)
        assert int(val_kn[0].sum()) == 6
        assert int(cross_kn[0].sum()) == 2


# ---------------------------------------------------------------------------
# 5.  Active QSGW application: rigid low valence, affine conduction
# ---------------------------------------------------------------------------

def _nontrivial_application_fit():
    """A fit whose valence regression is unmistakably not a rigid shift."""
    E = np.array([[-20.0, -6.0, -1.0, 1.0, 5.0, 12.0]])
    val = np.broadcast_to(
        (np.arange(E.shape[1]) < 3)[None, :], E.shape)
    # The trusted higher valence follows 1.6 E + 2, while conduction follows
    # 1.25 E - 0.75.  Only bands 1:5 enter the fit; bands 0 and 5 are the
    # low-valence/high-conduction extrapolation targets.
    E_qp = np.where(val, 1.6 * E + 2.0, 1.25 * E - 0.75)
    fit_mask = np.zeros_like(val)
    fit_mask[:, 1:5] = True
    fit = fit_scissor(
        E, E_qp, valence_mask_kn=val, fit_mask_kn=fit_mask,
        k_weights=full_bz_k_weights(E.shape[0]))
    assert fit.alpha_v == pytest.approx(1.6, abs=1e-14)
    assert fit.beta_v_ev == pytest.approx(2.0, abs=1e-14)
    assert fit.alpha_c == pytest.approx(1.25, abs=1e-14)
    assert fit.beta_c_ev == pytest.approx(-0.75, abs=1e-14)
    return E, val, fit


def test_active_qsgw_ignores_valence_regression_but_keeps_conduction_affine():
    E, val, fit = _nontrivial_application_fit()
    dfermi = 1.75
    out = qsgw_out_of_range_energies(
        E, fit, val, fermi_displacement_ev=dfermi)

    # LOW VALENCE: exactly the Fermi displacement.  The fitted valence law
    # predicts -30 eV for the -20 eV target, so this fails loudly if that
    # deliberately nontrivial regression ever leaks back into application.
    assert out[0, 0] == E[0, 0] + dfermi
    assert out[0, 0] == pytest.approx(-18.25, abs=0.0)
    assert out[0, 0] != pytest.approx(
        fit.alpha_v * E[0, 0] + fit.beta_v_ev, abs=1.0)

    # CONDUCTION: the existing fitted stretch/intercept is untouched.
    assert out[0, 5] == pytest.approx(
        fit.alpha_c * E[0, 5] + fit.beta_c_ev, abs=1e-14)
    assert out[0, 5] != pytest.approx(E[0, 5] + dfermi, abs=1e-3)


def test_candidate_fermi_change_is_applied_without_one_map_lag():
    E, val, fit = _nontrivial_application_fit()
    efermi_dft = 0.0
    # This map's protected/in-window frontier moves from DFT (-1,+1), whose
    # midgap is 0, to (0,+4), whose candidate midgap is +2 eV.
    H_candidate_diag = np.array([[-999.0, -5.0, 0.0, 4.0, 6.0, 999.0]])
    in_range = np.array([False, True, True, True, True, False])
    # Exact map wiring: fit once, put valence at DFT (shift=0) and conduction
    # at its FINAL affine value, apply the provisional partition, then solve
    # the candidate Fermi from that post-partition spectrum.
    provisional = qsgw_out_of_range_energies(
        E, fit, val, fermi_displacement_ev=0.0)
    H_probe_diag = np.where(
        in_range[None, :], H_candidate_diag, provisional)
    probe_eigenvalues = np.linalg.eigvalsh(np.diag(H_probe_diag[0]))
    efermi_candidate = 0.5 * (
        probe_eigenvalues[2] + probe_eigenvalues[3])
    assert efermi_candidate == 2.0

    scissor = qsgw_out_of_range_energies(
        E, fit, val,
        fermi_displacement_ev=efermi_candidate - efermi_dft)
    assert scissor[0, 5] == provisional[0, 5]
    final_diag = np.where(in_range[None, :], H_candidate_diag, scissor)
    final_eigenvalues = np.linalg.eigvalsh(np.diag(final_diag[0]))
    efermi_final = 0.5 * (final_eigenvalues[2] + final_eigenvalues[3])
    assert efermi_final == efermi_candidate

    # The out-of-range valence binding relative to THIS map's output Fermi is
    # exactly the DFT binding.  Anchoring to the entry/DFT Fermi would leave
    # it at -20 eV and fail this cell by the full +2 eV map displacement.
    assert final_diag[0, 0] - efermi_final == (
        E[0, 0] - efermi_dft)
    lagged_low_valence = E[0, 0] + (efermi_dft - efermi_dft)
    assert lagged_low_valence - efermi_candidate != E[0, 0] - efermi_dft

    # Protected/in-window diagonals still come from the full candidate H;
    # the high out-of-range conduction band still uses alpha_c/beta_c.
    assert np.array_equal(final_diag[:, 1:5], H_candidate_diag[:, 1:5])
    assert final_diag[0, 5] == pytest.approx(
        fit.alpha_c * E[0, 5] + fit.beta_c_ev, abs=1e-14)


def test_crossing_band_remains_identity_under_rigid_valence_policy():
    E, val, fit = _nontrivial_application_fit()
    crossing = np.zeros_like(val)
    crossing[:, 2] = True
    out = qsgw_out_of_range_energies(
        E, fit, val, fermi_displacement_ev=3.0,
        crossing_mask_kn=crossing)
    assert out[0, 2] == E[0, 2]


def test_sc_map_wires_the_same_map_candidate_fermi_not_the_entry_fermi():
    """Static seam gate: the pure policy must be fed F(H)'s own anchor."""
    source = (_SCISSOR.parent / "sc_iteration.py").read_text()
    start = source.index("def _apply_scissor_partition_policy(")
    stop = source.index("\ndef ", start + 1)
    block = source[start:stop]

    fit = block.index("_scissor_E_qp_for_outofrange(")
    probe = block.index("H_fermi_probe = apply_band_partition(")
    anchor = block.index("candidate_efermi_fn(H_fermi_probe)")
    displacement = block.index("fermi_displacement_ry = (")
    final_policy = block.index("qsgw_out_of_range_energies(")
    assert fit < probe < anchor < displacement < final_policy
    assert "candidate_efermi_ry - float(efermi_dft_ry)" in block
    assert "fermi_displacement_ry=0.0" in block
    # Conduction is already final in the probe and the same fitted object is
    # passed to the final policy; neither a second fit nor a copied affine law.
    assert block.count("_scissor_E_qp_for_outofrange(") == 1
    assert "scissor_E_qp_kn=scissor_provisional_ry" in block
    assert "scissor_fit," in block
    assert "solve_mp1_occupations(" not in block
    assert "requires the complete " in block
    assert "Fermi-crossing/frontier manifold inside the Sigma window" in block

    # The final H is independently re-anchored and refused if the tail moved
    # E_F, so the provisional non-circularity argument is executable.
    assert "final_efermi_ry = float(candidate_efermi_fn(H_partitioned))" in block
    assert "final_efermi_ry, candidate_efermi_ry" in block

    # The SC caller binds that callback to the candidate-spectrum owner; it
    # does not pass the map-entry Fermi level as a frozen scalar.
    map_start = source.index("def gw_iteration_map(")
    map_stop = source.index("def _scissor_E_qp_for_outofrange(", map_start)
    map_block = source[map_start:map_stop]
    assert "candidate_efermi_fn=lambda H: _partitioned_candidate_efermi(" in map_block


def test_frac_tol_is_validated():
    f = np.array([[1.0, 0.0]])
    with pytest.raises(ValueError, match="frac_tol"):
        classify_scissor_bands(f, frac_tol=0.9)


def test_band_classes_refuse_inverted_boundaries():
    with pytest.raises(ValueError, match="valence_stop"):
        ScissorBandClasses(valence_stop=5, conduction_start=2)

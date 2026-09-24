"""Owner ruling 2026-09-24: the sum-band tail law averages only states that
consume Sigma(E_nk).  A state on the Sigma(omega=0) fallback cannot move beta
(Fe 4^3: three such H-point states set a 14.9 meV tail shift, claim 2703).
Host NumPy, no device."""
import numpy as np

from gw.scissor import fit_scissor
from gw.sc_iteration import _fit_sum_band_tail


def _case():
    e_dft = np.array([[-5.0, -1.0, 2.0, 4.0, 6.0, 9.0],
                      [-4.5, -0.8, 2.5, 4.2, 6.5, 9.5]])
    shift = np.array([[-0.3, -0.2, 0.40, 0.50, 0.60, 0.70],
                      [-0.3, -0.2, 0.45, 0.55, 0.65, 2.57]])   # (1,5): Sigma(0) branch
    kw = dict(E_dft_kn_ev=e_dft, E_qp_kn_ev=e_dft + shift,
              valence_mask_kn=np.array([[1, 1, 0, 0, 0, 0]] * 2, dtype=bool),
              k_weights=np.array([1.0, 3.0]), conduction_rigid_mean=True)
    return kw, np.ones_like(e_dft, dtype=bool)


def test_no_fallback_is_the_old_fit_bitwise():
    kw, mask = _case()
    fit, n_ex, note = _fit_sum_band_tail(kw, mask, np.zeros_like(mask))
    assert fit == fit_scissor(fit_mask_kn=mask, **kw) and n_ex == 0 and note == ""


def test_a_sigma0_state_does_not_move_beta():
    kw, mask = _case()
    sigma0 = np.zeros_like(mask); sigma0[1, 5] = True
    fit, n_ex, _ = _fit_sum_band_tail(kw, mask, sigma0)
    assert n_ex == 1
    # Its Sigma(0)-branch energy is irrelevant: move it 1.87 eV, beta is bitwise equal.
    kw2 = dict(kw, E_qp_kn_ev=kw["E_qp_kn_ev"].copy()); kw2["E_qp_kn_ev"][1, 5] -= 1.87
    assert _fit_sum_band_tail(kw2, mask, sigma0)[0].beta_c_ev == fit.beta_c_ev
    # ... and equals the fit that never saw the state.
    assert fit == fit_scissor(fit_mask_kn=mask & ~sigma0, **kw)
    assert fit.beta_c_ev != fit_scissor(fit_mask_kn=mask, **kw).beta_c_ev


def test_no_qualifying_state_is_no_tail_law_never_nan():
    kw, mask = _case()
    fit, n_ex, note = _fit_sum_band_tail(kw, mask, ~kw["valence_mask_kn"])
    assert fit is None and n_ex == 8 and "E_DFT" in note


def test_unit_z_is_the_plain_mean_bitwise_and_a_satellite_barely_moves_beta():
    kw, mask = _case()
    none = np.zeros_like(mask)
    plain = _fit_sum_band_tail(kw, mask, none)[0]
    assert _fit_sum_band_tail(kw, mask, none, np.ones(mask.shape))[0] == plain
    # Owner 2026-09-24: Z-weighted.  One conduction state on a satellite:
    # shifted +3 eV with Z = 0.1 against Z = 0.8 elsewhere.
    z = np.full(mask.shape, 0.8); z[0, 4] = 0.1
    kw_sat = dict(kw, E_qp_kn_ev=kw["E_qp_kn_ev"].copy()); kw_sat["E_qp_kn_ev"][0, 4] += 3.0
    moved_mean = _fit_sum_band_tail(kw_sat, mask, none)[0].beta_c_ev - plain.beta_c_ev
    moved_z = (_fit_sum_band_tail(kw_sat, mask, none, z)[0].beta_c_ev
               - _fit_sum_band_tail(kw, mask, none, z)[0].beta_c_ev)
    assert moved_z < 0.2 * moved_mean          # measured 0.13: Z = 0.1 vs 0.8
    # Z outside (0, 1] is not a quasiparticle: the sample leaves the fit.
    z_bad = np.full(mask.shape, 0.8); z_bad[0, 4] = 1.7
    fit = _fit_sum_band_tail(kw_sat, mask, none, z_bad)[0]
    drop = mask.copy(); drop[0, 4] = False
    assert fit.n_fit_c == 7 and np.isclose(
        fit.beta_c_ev, _fit_sum_band_tail(kw_sat, drop, none, np.full(mask.shape, 0.8))[0].beta_c_ev)

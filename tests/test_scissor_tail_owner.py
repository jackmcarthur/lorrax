"""The reference scorer and SC map use the same rigid conduction-tail law."""
import numpy as np

from gw.scissor import fit_sum_band_tail


def _kwargs(energies, corrections, weights):
    return dict(E_dft_kn_ev=energies, E_qp_kn_ev=energies + corrections,
                valence_mask_kn=np.broadcast_to([True, False, False], energies.shape),
                k_weights=weights)


def test_star_expansion_preserves_tail_and_untrusted_samples_cannot_move_it():
    energies = np.array([[-1., 1., 2.], [-1., 1.5, 2.5]])
    shifts = np.array([[0., 1., 10.], [0., 3., 1e6]])
    stars = np.array([1., 3.])
    z = np.array([[1., .5, 2.], [1., 1., np.nan]])
    own_energy = np.array([[False, False, False], [False, False, True]])
    fit, excluded, note = fit_sum_band_tail(
        _kwargs(energies, shifts, stars), np.ones_like(energies, bool), own_energy, z)
    # Hand-calculated: (.5*1 + .5*10 + 3*3)/(.5 + .5 + 3).
    assert fit.beta_c_ev == 3.625
    assert fit.alpha_c == 1.
    assert excluded == 1 and note == ""
    expand = np.array([0, 1, 1, 1])
    full, _, _ = fit_sum_band_tail(
        _kwargs(energies[expand], shifts[expand], np.ones(4)),
        np.ones((4, 3), bool), own_energy[expand], z[expand])
    assert full.beta_c_ev == fit.beta_c_ev
    assert full.w_fit_c == fit.w_fit_c


def test_first_update_is_unit_weight_and_empty_training_has_no_tail_law():
    energies = np.array([[-1., 1., 2.]])
    shifts = np.array([[0., 2., 4.]])
    kwargs = _kwargs(energies, shifts, np.ones(1))
    fit, _, _ = fit_sum_band_tail(kwargs, np.ones((1, 3), bool), np.zeros((1, 3), bool))
    assert fit.beta_c_ev == 3.
    fit, _, note = fit_sum_band_tail(
        kwargs, np.ones((1, 3), bool), np.array([[False, True, True]]))
    assert fit is None
    assert "no conduction state" in note

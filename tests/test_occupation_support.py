"""A band is in a Green's-function branch iff its weight is at least 1e-5.

The branch weight is ``f`` on the occupied side and ``1 - f`` on the empty
side.  ``gw.efermi.band_in_occupation_window`` keeps it iff ``|w| >= 1e-5``:
for Fermi-Dirac, ``|E - mu| <= 11.5 kBT``.  The one-shot and every SC map
use the same predicate, so identical occupations give identical branch
support. These tests do not bound the complete GW energy change or certify
bitwise equality of the full map. The retired deck key
``occupation_window_threshold`` (a 0.005 weight floor by default) refuses.
"""

import numpy as np
import pytest

import jax.numpy as jnp

from gw import efermi
from gw.efermi import OCCUPATION_WEIGHT_FLOOR, band_in_occupation_window
from gw.mpa import sigma_windows as SW
from gw.ppm_windows import _SigmaBranch, branches_for_omega_grid


_OMEGA = np.asarray([0.0, 0.25, 0.5])
_IDX = np.arange(_OMEGA.size)


def _branch(energies, weights, *, space="val"):
    E_A = jnp.asarray(np.asarray(energies, dtype=np.float64)[None, :])
    bw = (None if weights is None
          else jnp.asarray(np.asarray(weights, dtype=np.float64)[None, :]))
    return _SigmaBranch("pos_" + space, E_A, jnp.ones_like(E_A, dtype=bool),
                        space, False, _OMEGA, _IDX, band_weight=bw)


def _support(branch):
    mask, bounds = SW._a_space(branch, lambda E: np.ones(E.shape, bool))
    return np.asarray(mask)[0], bounds


def _branch_masks(f):
    f = jnp.asarray(np.asarray(f, dtype=np.float64)[None, :])
    E = jnp.zeros_like(f)
    branches = branches_for_omega_grid(
        np.asarray([0.0, 0.25]), E_cond=E, H_val=-E,
        cond_mask=(f != 1.0), val_mask=(f != 0.0),
        cond_weight=1.0 - f, val_weight=f)
    got = {b.space: np.asarray(b.base_mask_A)[0] for b in branches}
    return got["cond"], got["val"]


def test_the_floor_is_1e_minus_5():
    floor = OCCUPATION_WEIGHT_FLOOR
    assert floor == 1e-5
    w = np.asarray([0.0, np.nextafter(floor, 0.0), floor, 2.0 ** -53,
                    -floor, 0.5, 1.0])
    np.testing.assert_array_equal(band_in_occupation_window(w),
                                  [False, False, True, False, True, True, True])


def test_the_conduction_weight_has_the_same_floor():
    """``1.0 - f`` below 1e-5 leaves the empty branch, as ``f`` does the occupied."""
    f = np.asarray([0.5, 1.0 - 2e-5, 1.0 - 5e-6, 1.0 - 2.0 ** -53, 1.0])
    u = 1.0 - f
    np.testing.assert_array_equal(band_in_occupation_window(u),
                                  [True, True, False, False, False])


def test_fermi_dirac_support_ends_at_11_5_kbt_on_both_sides():
    """|w| >= 1e-5 iff |E - mu| <= kBT ln(1e5 - 1) = 11.51 kBT."""
    kbt = 0.02
    E = np.asarray([-11.6, -11.4, 0.0, 11.4, 11.6]) * kbt
    f = np.asarray(efermi.fd_occupations(E[None, :], 0.0, kbt))[0]
    cond, val = _branch_masks(f)
    np.testing.assert_array_equal(val, [True, True, True, True, False])
    np.testing.assert_array_equal(cond, [False, True, True, True, True])


def test_the_claims_2793_pair_is_kept_and_the_float64_tail_dropped():
    """The CLAIMS 2793 pair straddled 0.005; both belong to the branch now.
    Weights below 1e-5, which the float64 floor 2**-53 kept, do not."""
    f = [0.00499944814812, 0.00500410083235, 2e-5, 5e-6, 1e-15, 0.0]
    _cond, val = _branch_masks(f)
    np.testing.assert_array_equal(val, [True, True, True, False, False, False])


def test_negative_mp1_weights_are_kept_and_zeros_excluded():
    keep, bounds = _support(_branch([-0.2, -0.1, 0.1, 0.2],
                                    [-0.0355, 0.0, 1e-17, 0.3]))
    np.testing.assert_array_equal(keep, [True, False, False, True])
    assert bounds == (-0.2, 0.2)


def test_an_insulating_branch_has_no_weight_and_is_untouched():
    keep, bounds = _support(_branch([0.1, 0.2, 0.3], None))
    assert keep.all() and bounds == (0.1, 0.3)


def test_chi_supports_use_the_same_predicate():
    from gw.w_isdf import _occupation_support_slices
    occ = np.asarray([[1.0, 1.0 - 1e-6, 0.4, 1e-4, 1e-6, 0.0]])
    f_slice, u_slice = _occupation_support_slices(occ)
    assert f_slice == slice(0, 4)
    assert u_slice == slice(2, 6)
    gapped = np.asarray([[1.0, 1.0, 0.0, 0.0]])
    assert _occupation_support_slices(gapped) == (slice(0, 2), slice(2, 4))


def test_there_is_one_predicate():
    from gw import ppm_windows, response_bank, w_isdf
    for mod in (ppm_windows, w_isdf, SW, response_bank):
        assert mod.band_in_occupation_window is band_in_occupation_window


def test_the_response_bank_samples_the_same_support():
    from gw.response_bank import response_sample_weights
    f = np.asarray([1.0, 1.0 - 1e-6, 0.5, 1e-4, 1e-6, 0.0])
    ft, ut, receipt = response_sample_weights(f, 1.0 - f)
    np.testing.assert_array_equal(ft != 0, [True, True, True, True, False, False])
    np.testing.assert_array_equal(ut != 0, [False, False, True, True, True, True])
    np.testing.assert_array_equal(ft != 0, band_in_occupation_window(f))
    np.testing.assert_array_equal(ut != 0, band_in_occupation_window(1.0 - f))
    assert receipt["occupation_activity_floor"] == OCCUPATION_WEIGHT_FLOOR


def test_the_retired_key_refuses_by_name(tmp_path):
    from gw.gw_config import LorraxConfig
    deck = tmp_path / "gw.in"
    deck.write_text("[cohsex]\nsys_dim = 3\ncompute_mode = mpa\n"
                    "sigma_w_model = shared_pole\nnval = 4\nncond = 20\n"
                    "number_bands = 40\noccupation_window_threshold = 0.995\n")
    with pytest.raises(ValueError, match="occupation_window_threshold"):
        LorraxConfig.from_input_file(str(deck), print_fn=lambda _: None)


def test_the_static_occupation_projector_is_not_thresholded():
    """``cohsex_sigma.build_Gij`` weights every Sigma band by f and drops none:
    a cut would delete electrons from the Hartree density."""
    import inspect
    from gw.cohsex_sigma import build_Gij

    src = inspect.getsource(build_Gij)
    assert "band_in_occupation_window" not in src
    assert "Gij[:, idx, idx] = f_win" in src

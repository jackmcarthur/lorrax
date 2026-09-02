from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp

from gw.mpa.sigma import _batch_rows, _branches
from gw.ppm_sigma import _SigmaPhysicsState, _ppm_as_one_pole_store_fields


def _row(poles):
    return SimpleNamespace(
        pole_indices=np.asarray(poles, np.int32),
        bounds=np.zeros((len(poles), 6)),
        phase_real=np.zeros(len(poles), bool))


def test_batch_rows_relocalize_pole_selection_per_batch():
    shared = _row(range(8))
    low_only = _row((0, 1))
    np.testing.assert_array_equal(
        _batch_rows(shared, range(4, 8))[0], np.arange(4, dtype=np.int32))
    assert _batch_rows(low_only, range(4, 8)) is None


def test_ppm_one_pole_fields_use_the_mpa_residue_normalization():
    """PPM Wc(0) = -2B/Omega enters the MPA model without rescaling."""
    Omega = jnp.asarray([[[0.7, 0.8], [0.9, 1.0]]])
    B = jnp.asarray([[[1.0 + 0.2j, -0.4], [0.3j, 0.6 - 0.1j]]])
    live = jnp.asarray([[[True, False], [True, False]]])
    state = _SigmaPhysicsState(
        efermi=jnp.asarray(0.0),
        E_cond=jnp.zeros((1, 1)), H_val=jnp.zeros((1, 1)),
        cond_mask=jnp.ones((1, 1), bool), val_mask=jnp.ones((1, 1), bool),
        B_corr=B, Omega_abs=Omega, B_mask=live,
        invalid_mask=~live, n_total_modes=jnp.asarray(4),
        n_invalid=jnp.asarray(2))
    Omega_p, B_p = map(np.asarray, _ppm_as_one_pole_store_fields(state))

    assert Omega_p.shape == B_p.shape == (1, 1, 2, 2)
    np.testing.assert_array_equal(B_p[0][~np.asarray(live)], 0.0)
    np.testing.assert_array_equal(Omega_p[0][~np.asarray(live)], 0.0)
    mask = np.asarray(live)
    ppm_wc0 = -2.0 * np.asarray(B)[mask] / np.asarray(Omega)[mask]
    mpa_wc0 = -2.0 * B_p[0][mask] / Omega_p[0][mask]
    np.testing.assert_array_equal(mpa_wc0, ppm_wc0)


def test_small_gap_branching_follows_occupation_not_energy_sign():
    wfns = SimpleNamespace(
        enk=np.asarray([[-0.1, -0.01, 0.02, 0.3]]),
        occ=np.asarray([[1.0, 0.0, 1.0, 0.0]]),
        # ``sigma_sum`` and DELIBERATELY NOT ``full``.  Since the chi/Sigma
        # split those are different windows -- ``full`` is the LOADED extent
        # max(chi, sigma), ``sigma_sum`` is the band sum these branches run
        # over -- and the causal branching is a statement about the SIGMA
        # sum.  Omitting ``full`` makes this cell fail loudly if the
        # production code ever reaches back for the larger consumer's count.
        slices=SimpleNamespace(sigma_sum=slice(None)))
    branches = _branches(wfns, np.asarray([-0.2, 0.4]), 0.0)
    pos_cond = next(b for b in branches
                    if b.space == "cond" and not b.neg_omega_half)
    neg_val = next(b for b in branches
                   if b.space == "val" and b.neg_omega_half)
    assert pos_cond.base_mask_A.tolist() == [[False, True, False, True]]
    assert neg_val.base_mask_A.tolist() == [[True, False, True, False]]
    assert float(pos_cond.E_A[0, 1]) < 0.0
    assert float(neg_val.E_A[0, 2]) < 0.0


def test_branches_metallize_only_with_an_occupation_state():
    """None ⇒ incumbent occ>0.5 masks (no weights); a state ⇒ exact supports
    with unclipped (f, 1−f) weights signed against the state's mu."""
    from types import SimpleNamespace
    import jax.numpy as jnp
    from gw.mpa.sigma import _branches

    enk = np.asarray([[0.1, 0.2, 0.3]])
    occ = np.asarray([[1.0, 0.6, 0.0]])
    wfns = SimpleNamespace(
        enk=jnp.asarray(enk), occ=jnp.asarray(occ),
        # ``sigma_sum`` and DELIBERATELY NOT ``full`` — the same stub the
        # cell above carries, and for the same reason (18f4baa3).  This cell
        # arrived with the metal branch, where ``_branches`` still read
        # ``full``; after the chi/Sigma split that name is the LOADED extent
        # max(chi, sigma) and the causal branching is a statement about the
        # SIGMA band sum.  Omitting ``full`` makes the cell fail loudly if
        # the production code reaches back for the larger consumer's count.
        slices=SimpleNamespace(sigma_sum=slice(0, 3)))
    omega = np.asarray([0.0, 0.4])

    legacy = _branches(wfns, omega, 0.25)
    assert all(b.band_weight is None for b in legacy)
    np.testing.assert_array_equal(
        np.asarray(legacy[0].base_mask_A), occ <= 0.5)

    state = SimpleNamespace(
        f_kn=np.asarray([[1.0, 0.7, -0.02]]), mu_ry=0.25,
        n_electrons=1.68, occ_hash="h")
    metal = _branches(wfns, omega, 0.25, occupation_state=state)
    cond, val = metal[0], metal[1]
    np.testing.assert_array_equal(np.asarray(cond.base_mask_A),
                                  [[False, True, True]])
    np.testing.assert_array_equal(np.asarray(val.base_mask_A),
                                  [[True, True, True]])
    np.testing.assert_allclose(np.asarray(cond.band_weight),
                               [[0.0, 0.3, 1.02]])
    np.testing.assert_allclose(np.asarray(val.band_weight),
                               [[1.0, 0.7, -0.02]])
    np.testing.assert_allclose(np.asarray(cond.E_A), enk - 0.25)

    try:
        _branches(wfns, omega, 0.30, occupation_state=state)
    except ValueError as err:
        assert "inconsistent" in str(err)
    else:
        raise AssertionError("mu mismatch was not refused")

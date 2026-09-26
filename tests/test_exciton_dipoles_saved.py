"""Per-state exciton dipoles saved beside the eigenvectors (``exciton_data/dipoles``).

What is pinned, on a synthetic Hermitian block with complex eigenvectors:

  * the distributed contraction (``exciton_dipoles_distributed``, the jax arm of
    ``exciton_dipole_projections``) is the NumPy contraction, state for state;
  * ``write_eigenvectors_stream(dipoles=...)`` stores exactly those numbers,
    ``load_exciton_dipoles_h5`` reads them back, and the pad rows the writer
    trims contribute nothing to them;
  * ε₂ built from the SAVED dipoles and eigenvalues
    (``eps2_from_exciton_dipoles``) is the resolvent spectrum
    ``−pref·Im⟨d|(ω + iη − H)⁻¹|d⟩/π`` that ``absorption_haydock`` continues
    fractions to — the same prefactor, the same Lorentzian;
  * a non-TDA write or a wrong-shaped dipole array is refused before any file exists.
"""
from __future__ import annotations

import numpy as np
import pytest

from bse.absorption_common import (
    RYD2EV,
    eps2_from_exciton_dipoles,
    exciton_dipole_projections,
    exciton_dipoles_distributed,
    load_exciton_dipoles_h5,
)
from bse.bse_io import write_eigenvectors_stream

N_COND, N_VAL = 3, 2            # logical window, NOT square in (c, v)
NC_PAD, NV_PAD = 4, 2           # the solver's padded extents
NKX, NKY, NKZ = 2, 2, 1
NK = NKX * NKY * NKZ
N_T = N_COND * N_VAL * NK
V_CELL, N_SPIN, N_SPINOR = 250.0, 1, 2
SEED = 20260926


def _case():
    """H over the logical block; eigenvectors and dipole in the padded solver layout."""
    rng = np.random.default_rng(SEED)
    M = rng.standard_normal((N_T, N_T)) + 1j * rng.standard_normal((N_T, N_T))
    H = 0.05 * (M + M.conj().T) + np.diag(np.linspace(0.1, 0.6, N_T))
    E, V = np.linalg.eigh(H)
    A = np.zeros((N_T, 1, NC_PAD, NV_PAD, NK), complex)
    A[:, 0, :N_COND, :N_VAL, :] = V.T.reshape(N_T, N_COND, N_VAL, NK)
    d = np.zeros((3, NC_PAD, NV_PAD, NK), complex)
    d_flat = rng.standard_normal((3, N_T)) + 1j * rng.standard_normal((3, N_T))
    d[:, :N_COND, :N_VAL, :] = d_flat.reshape(3, N_COND, N_VAL, NK)
    return H, E, A, d, d_flat


def test_the_distributed_contraction_is_the_numpy_one():
    import jax.numpy as jnp
    _, _, A, d, _ = _case()
    want = exciton_dipole_projections(A[:, 0], d)
    got = exciton_dipoles_distributed(jnp.asarray(A), d, N_T)
    assert got.shape == (N_T, 3) and got.dtype == np.complex128
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-13)


def test_saved_dipoles_round_trip_and_reproduce_the_resolvent(tmp_path):
    import jax.numpy as jnp
    H, E, A, d, d_flat = _case()
    dip = exciton_dipoles_distributed(jnp.asarray(A), d, N_T)
    out = tmp_path / "eigenvectors.h5"
    write_eigenvectors_stream(str(out), E, A, N_VAL, N_COND, NKX, NKY, NKZ,
                              N_T, dipoles=dip)
    E_file, D_file = load_exciton_dipoles_h5(str(out))
    np.testing.assert_allclose(D_file, dip, rtol=0, atol=0)
    np.testing.assert_allclose(E_file, E, rtol=1e-12)

    eta = 0.01
    w = np.linspace(0.0, 0.8, 41)
    eps2 = eps2_from_exciton_dipoles(w, E_file, D_file, eta, V_CELL, NK,
                                     N_SPIN, N_SPINOR)
    pref = 16.0 * np.pi ** 2 / (V_CELL * NK * N_SPIN * N_SPINOR)
    res = np.empty((w.size, 3))
    for i, om in enumerate(w):
        X = np.linalg.solve((om + 1j * eta) * np.eye(N_T) - H, d_flat.T)
        res[i] = -pref * np.einsum("aT,Ta->a", d_flat.conj(), X).imag / np.pi
    np.testing.assert_allclose(eps2, res, rtol=1e-10, atol=1e-12 * res.max())


@pytest.mark.parametrize("kind", ["non_tda", "shape"])
def test_bad_dipoles_are_refused_before_the_file(tmp_path, kind):
    _, E, A, _, _ = _case()
    out = tmp_path / "eigenvectors.h5"
    if kind == "non_tda":
        vecs = np.concatenate([A, A], axis=1)
        kw = dict(use_tda=False, dipoles=np.zeros((N_T, 3), complex))
    else:
        vecs, kw = A, dict(dipoles=np.zeros((N_T - 1, 3), complex))
    with pytest.raises(ValueError, match="dipoles must be"):
        write_eigenvectors_stream(str(out), E, vecs, N_VAL, N_COND,
                                  NKX, NKY, NKZ, N_T, **kw)
    assert not out.exists()


def test_a_file_without_dipoles_says_how_to_get_them(tmp_path):
    _, E, A, _, _ = _case()
    out = tmp_path / "eigenvectors.h5"
    write_eigenvectors_stream(str(out), E, A, N_VAL, N_COND, NKX, NKY, NKZ, N_T)
    with pytest.raises(KeyError, match="--dipole"):
        load_exciton_dipoles_h5(str(out))


def test_ryd2ev_is_the_writers_constant():
    """The file stores eV; the loader must undo the writer's own factor."""
    assert abs(RYD2EV - 13.6056980659) < 1e-9

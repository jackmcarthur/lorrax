"""``densified_dipoles(full_window=True)``: every fine-window pair (CPU, 2x2 host mesh).

The TD-aGW system builder needs the velocity over the whole window (the vv'
and cc' blocks as well as c <- v).  On the toy tight-binding slab of
``tests/test_bse_kgrid_dipoles_cpu.py`` (imported, not copied):

1. The default outputs are bitwise those of a ``full_window=True`` run: the
   flag adds ``v_window`` / ``e_dft_window`` / ``r_window`` (and the pair rule
   it used) and changes nothing else.
2. The window's c <- v block over its own ΔE IS ``d_alpha`` and its energy
   differences ARE ``de_cv`` (bitwise: one contraction, two slices).
3. The window is Hermitian, and its off-diagonal pairs (the ones the TD
   builder turns into r = v/(iΔE); the diagonal band velocities are not
   used) match the exact velocity of the toy in the densified states' own
   gauge to the htransform's interpolation error; the bra/ket-swapped
   window (the red twin) does not.

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

_TOY = pathlib.Path(__file__).with_name("test_bse_kgrid_dipoles_cpu.py")
_spec = importlib.util.spec_from_file_location("_bse_kgrid_dipoles_toy", _TOY)
toy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(toy)

from bandstructure.bse_setup import compute_wfns_fi  # noqa: E402
from bse.bse_densify import densified_dipoles  # noqa: E402
from ffi.fft import xla_reference  # noqa: E402

NV, LO, HI = toy.N_VAL, toy.B_MIN, toy.B_MAX


def _run(mesh, c, e, v, *, full_window):
    kc = toy._grid(toy.KC)
    with xla_reference():
        fine = compute_wfns_fi(
            ctilde=jnp.asarray(c), B_at_mu=toy._nodes(c),
            enk_sigma=jnp.asarray(e.T), kgrid_co=toy.KC, kgrid_fi=toy.KF,
            band_window_fi=(LO, HI), mesh_xy=mesh, return_coeffs=True)
        dip = densified_dipoles(
            (v, e, "toy dH/dk", "toy.h5"), sym=toy._sym(kc), fh_energies=e,
            basis=SimpleNamespace(ctilde=jnp.asarray(c), band_range=(0, toy.NB),
                                  basis_at_nodes=toy._nodes(c)),
            fine=fine, coarse_grid=toy.KC, n_val=NV, window_start=LO,
            a_band_index=None, mesh_xy=mesh, log_fn=lambda *a: None,
            full_window=full_window)
    return dip, fine


@pytest.fixture(scope="module")
def runs():
    mesh, h = toy._mesh(), toy._model()
    c, e, v = toy._states(h, toy._grid(toy.KC), phases=np.exp(
        2j * np.pi * np.random.default_rng(11).random((np.prod(toy.KC), toy.NB))))
    d0, _ = _run(mesh, c, e, v, full_window=False)
    d1, fine = _run(mesh, c, e, v, full_window=True)
    return h, d0, d1, fine


def test_default_outputs_are_bitwise_unchanged(runs):
    _, d0, d1, _ = runs
    assert "v_window" not in d0 and "e_dft_window" not in d0
    assert set(d1) - set(d0) == {"v_window", "e_dft_window", "r_window",
                                 "pair_rule", "pair_gap_ev"}
    for key, val in d0.items():
        if isinstance(val, np.ndarray):
            assert val.dtype == d1[key].dtype and np.array_equal(val, d1[key]), key
        else:
            assert val == d1[key], key


def test_window_holds_the_cv_dipoles_bitwise(runs):
    _, _, d1, _ = runs
    vw, ew = d1["v_window"], d1["e_dft_window"]
    nq = np.prod(toy.KF)
    assert vw.shape == (3, nq, HI - LO, HI - LO) and ew.shape == (nq, HI - LO)
    de = ew[:, NV:, None] - ew[:, None, :NV]
    assert np.array_equal(de, d1["de_cv"])
    assert np.array_equal(vw[:, :, NV:, :NV] / de[None], d1["d_alpha"])


def test_window_is_hermitian_and_exact_and_the_swap_is_not(runs):
    h, _, d1, fine = runs
    vw = d1["v_window"]
    herm = np.abs(vw - np.conj(np.swapaxes(vw, -1, -2))).max() / np.abs(vw).max()
    assert herm < 1e-12
    # Exact velocity at the fine q, rotated into the densified states' gauge.
    ce, _, ve = toy._states(h, fine.q_frac)
    w = np.einsum("qna,qam->qnm", ce[:, LO:HI].conj(), np.asarray(fine.coeffs_fi))
    exact = np.einsum("qnm,qxnl,qlk->xqmk", w.conj(), ve[:, :, LO:HI, LO:HI], w)
    off = ~np.eye(HI - LO, dtype=bool)
    scale = np.abs(exact[:2]).max()
    err = np.abs((vw[:2] - exact[:2])[..., off]).max() / scale
    swap = np.abs((np.swapaxes(vw, -1, -2)[:2] - exact[:2])[..., off]).max() / scale
    # The htransform's own interpolation error on this 6x6 toy is ~1%
    # (1.4% off the diagonal; the interpolated band velocities on the
    # diagonal reach 2.0%).
    assert err < 0.02 and swap > 10 * err

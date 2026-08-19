"""The SC active window is [b0, b3) GLOBAL; its width alone is not a window.

Two defects of the same shape, both silent:

1. ``_dft_psi_sphere`` asked ``WfnLoader.load`` for ``bands=(0, nb_sigma)``.
   ``load`` indexes the FILE's bands (``wfn_loader.py:1158-1165``) while
   ``nb_sigma = b3 − b0`` is a width, so the read was the right number of
   the WRONG BANDS whenever ``b0 != 0``.  V_H is O(400 Ry) and the band
   COUNT was still right, so ``rho_from_wfns``'s electron-count check —
   which verifies the count it was handed — could not see it.
2. Every occupancy in ``sc_iteration`` (``val_mask_active``, ``n_occ``,
   the midgap ``E[:, :n_occ]``, the ``fermi_level_step`` target) is
   ``meta.nelec``, a count from band 0, indexed into the window.  Correct
   only at ``b0 == 0``, so ``run_sc_driver`` refuses anything else.

Requires jax (``sc_iteration`` imports it at module scope), so this runs
in the container, not on a login node.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

jax = pytest.importorskip("jax")

import numpy as np                                            # noqa: E402

from gw import sc_iteration                                   # noqa: E402
from gw.wavefunction_bundle import BandSlices                 # noqa: E402


class _RecordingWfn:
    """Records the band window ``_dft_psi_sphere`` asks for."""

    def __init__(self, nk=2, ngkmax=8):
        self.calls = []
        self.nk = nk
        self.ngkmax = ngkmax

    def load(self, *, bands, k, sharding=None, bispinor=False):
        self.calls.append((int(bands[0]), int(bands[1])))
        nb = int(bands[1]) - int(bands[0])
        return jax.numpy.zeros(
            (self.nk, nb, 1, self.ngkmax), dtype=jax.numpy.complex128)

    def box_index(self, *, k):
        return np.zeros((self.nk, self.ngkmax, 3), dtype=np.int32)


class _Inputs:
    def __init__(self, band_slices, nb_carry, nk=2):
        self.band_slices = band_slices
        self.wfn = _RecordingWfn(nk=nk)
        self.kin_ion_dft = jax.numpy.zeros(
            (nk, nb_carry, nb_carry), dtype=jax.numpy.complex128)


def _slices(b0, b3):
    return BandSlices.from_band_edges(b0, b0, b0, b3, b3)


def test_psi_sphere_reads_the_global_sigma_window():
    """b0 != 0: the read must start at b0, not at 0."""
    sc_iteration._PSI_G_CACHE.clear()
    bs = _slices(12, 44)                       # 32 active bands starting at 12
    inp = _Inputs(bs, nb_carry=32)
    sc_iteration._dft_psi_sphere(inp)
    assert inp.wfn.calls == [(12, 44)], inp.wfn.calls
    sc_iteration._PSI_G_CACHE.clear()


def test_psi_sphere_is_unchanged_at_b0_zero():
    """The production decks in use have b0 = 0; that path must not move."""
    sc_iteration._PSI_G_CACHE.clear()
    inp = _Inputs(_slices(0, 128), nb_carry=128)
    sc_iteration._dft_psi_sphere(inp)
    assert inp.wfn.calls == [(0, 128)]
    sc_iteration._PSI_G_CACHE.clear()


def test_psi_sphere_cache_key_separates_equal_width_windows():
    """Two windows of the same width at different b0 are different psi."""
    sc_iteration._PSI_G_CACHE.clear()
    wfn = _RecordingWfn()
    a = _Inputs(_slices(0, 32), nb_carry=32)
    b = _Inputs(_slices(32, 64), nb_carry=32)
    a.wfn = b.wfn = wfn                        # same loader, same id()
    sc_iteration._dft_psi_sphere(a)
    sc_iteration._dft_psi_sphere(b)
    assert wfn.calls == [(0, 32), (32, 64)], wfn.calls
    sc_iteration._PSI_G_CACHE.clear()


def test_psi_sphere_refuses_a_width_that_disagrees_with_the_carry():
    """The b0-relative vs global mismatch itself, caught before the read."""
    sc_iteration._PSI_G_CACHE.clear()
    inp = _Inputs(_slices(12, 44), nb_carry=44)      # 44 != 44-12
    with pytest.raises(ValueError, match="b0-relative"):
        sc_iteration._dft_psi_sphere(inp)
    assert inp.wfn.calls == []
    sc_iteration._PSI_G_CACHE.clear()


def test_run_sc_driver_refuses_b0_nonzero():
    """The occupancy assumption, refused rather than silently applied."""
    import types

    bs = _slices(12, 44)
    meta = types.SimpleNamespace(nelec=26, nspinor=1, kgrid=(1, 1, 1))
    with pytest.raises(NotImplementedError, match="b0=12"):
        sc_iteration.run_sc_driver(
            None, None, None,
            quad=None, e_ref=0.0, static_head_terms=None,
            head_resolver=None, config=None, meta=meta,
            mesh_xy=None, sym=None, wfn=None, centroid_indices=None,
            band_slices=bs, input_dir=".",
            tensors_filename="unused-before-b0-refusal.h5",
            enk_dft=np.zeros((2, 32)), print_fn=lambda *a, **k: None)

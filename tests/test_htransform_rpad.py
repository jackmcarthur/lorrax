"""Exact-zero r padding for the htransform Galerkin Q accumulator."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
import pytest

from common import wfn_transforms as wt


def test_cri3_carried_extent_uses_the_full_4x4_mesh():
    mesh = SimpleNamespace(size=16, shape={"x": 4, "y": 4})
    carried = wt.flat_r_carried_extent(1_406_250, 2, mesh)
    assert carried == 1_406_256
    assert carried % 4 == 0
    assert (2 * carried) % 16 == 0


def test_iter_rpad_appends_zeros_after_the_physical_slice(monkeypatch):
    if len(jax.devices()) >= 4:
        devices = np.asarray(jax.devices()[:4]).reshape(2, 2)
    else:
        devices = np.asarray(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(devices, ("x", "y"))
    meta = SimpleNamespace(
        nk_tot=1, nspinor=1, kgrid=(1, 1, 1), fft_grid=(1, 1, 3))

    class _Sym:
        kvecs_asints = np.zeros((1, 3), dtype=np.int32)

    class _Loader:
        def box_index(self, *, k):
            assert k == "full_bz"
            return np.zeros((1, 1), dtype=np.int32)

        def symmetry(self):
            return _Sym()

    monkeypatch.setattr(
        wt, "_slice_bands_gflat", lambda psi, lo, width, mesh: psi)

    def _physical_slice(*args, **kwargs):
        return jnp.asarray([[[[1.0, 2.0, 3.0]]]], dtype=jnp.complex128)

    monkeypatch.setattr(wt, "to_rchunk", _physical_slice)
    psi_g = jnp.ones((1, 1, 1, 1), dtype=jnp.complex128)
    chunks = list(wt.iter_psi_rchunk_bandwise(
        _Loader(), None, meta, mesh, (0, 1), 0, 3, False,
        band_chunk_ranges=[(0, 1)], r_pad_to=4, psi_G_flat=psi_g))

    assert len(chunks) == 1
    np.testing.assert_array_equal(
        np.asarray(chunks[0][1]),
        np.asarray([[[[1.0, 2.0, 3.0, 0.0]]]], dtype=np.complex128))


def test_iter_rpad_refuses_to_truncate_physical_cells():
    meta = SimpleNamespace(nk_tot=1)
    with pytest.raises(ValueError, match="r_pad_to must be at least"):
        next(wt.iter_psi_rchunk_bandwise(
            object(), None, meta, object(), (0, 1), 0, 3, False,
            r_pad_to=2))

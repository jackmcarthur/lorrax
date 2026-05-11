"""Tests for ``file_io.zeta_reader.ZetaReader``.

We exercise the header-reading + IBZ-detection surface against a
synthetic ``zeta_q.h5`` built with the same helpers the writer uses
(``copy_mf_header`` + ``write_isdf_header`` + a hand-rolled ``zeta_q``
dataset).  The G-flat read path requires a real jax mesh + FFT plan
and isn't easily unit-testable here — it's covered by the end-to-end
V_q smoke run.
"""
from __future__ import annotations

import os

import h5py
import jax
import numpy as np
import pytest
from jax.sharding import Mesh

from file_io.mf_header import copy_mf_header
from file_io.isdf_header import IsdfHeader, write_isdf_header
from file_io.zeta_reader import ZetaReader

# Reuse the synthetic-WFN helper from the header tests.
from tests.test_mf_isdf_header_roundtrip import _make_fake_wfn


def _build_zeta_h5(tmp_path, n_q_disk: int, n_rtot: int = 8, n_rmu: int = 4,
                   density: str = "scalar", vertex_mu_L: int = 0):
    """Lay out a synthetic ``zeta_q.h5`` with mf_header + isdf_header +
    a zero-filled ``zeta_q`` dataset of the requested shape."""
    wfn_path = str(tmp_path / "WFN.h5")
    out_path = str(tmp_path / "zeta_q.h5")
    _make_fake_wfn(wfn_path)
    copy_mf_header(wfn_path, out_path, dst_mode='w')

    hdr = IsdfHeader.build(
        r_mu_fft_idx=np.arange(3 * n_rmu, dtype=np.int32).reshape(n_rmu, 3) % 8,
        fft_grid=(8, 8, 8),
        density=density,
        vertex_mu_L=vertex_mu_L,
    )
    write_isdf_header(out_path, hdr, mode='a')

    with h5py.File(out_path, 'a') as f:
        f.create_dataset('zeta_q',
                          shape=(n_q_disk, n_rtot, n_rmu),
                          dtype=np.complex128)
    return out_path


@pytest.fixture
def single_device_mesh():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                 axis_names=('x', 'y'))


def test_zeta_reader_loads_mf_and_isdf_headers(tmp_path, single_device_mesh):
    out_path = _build_zeta_h5(tmp_path, n_q_disk=12,
                              n_rtot=8, n_rmu=4)
    with ZetaReader(out_path, mesh=single_device_mesh) as zr:
        # mf_header surface
        assert zr.nkpts == 3      # from _make_fake_wfn
        assert zr.nbands == 4
        assert zr.nspin == 1
        assert zr.nspinor == 2
        np.testing.assert_array_equal(zr.kgrid, [2, 3, 4])
        assert zr.fft_grid.shape == (3,)
        assert zr.ntran == 8

        # isdf_header surface
        assert zr.density == 'scalar'
        assert zr.vertex_mu_L == 0
        assert zr.n_rmu == 4
        assert zr.r_mu_fft_idx.shape == (4, 3)
        assert zr.r_mu_crystal.shape == (4, 3)

        # Disk-shape detection
        assert zr.n_q_on_disk == 12
        assert zr.n_rtot_disk == 8
        assert zr.n_rmu_disk == 4


def test_zeta_reader_detects_ibz_q_axis(tmp_path, single_device_mesh):
    """An IBZ-only zeta_q.h5 has a smaller leading axis; ZetaReader
    exposes that via ``n_q_on_disk``."""
    out_path = _build_zeta_h5(tmp_path, n_q_disk=4, n_rtot=8, n_rmu=4)
    with ZetaReader(out_path, mesh=single_device_mesh) as zr:
        assert zr.n_q_on_disk == 4   # IBZ subset of 2*3*4 = 24 full BZ
        # Full BZ size derivable from kgrid:
        full_bz = int(np.prod(zr.kgrid))
        assert full_bz == 24
        assert zr.n_q_on_disk < full_bz


def test_zeta_reader_vertex_mu_L_is_int(tmp_path, single_device_mesh):
    out_path = _build_zeta_h5(tmp_path, n_q_disk=4, density='current',
                               vertex_mu_L=2)
    with ZetaReader(out_path, mesh=single_device_mesh) as zr:
        assert zr.density == 'current'
        assert zr.vertex_mu_L == 2
        assert isinstance(zr.vertex_mu_L, int)


def test_zeta_reader_context_manager_closes(tmp_path, single_device_mesh):
    out_path = _build_zeta_h5(tmp_path, n_q_disk=4)
    zr = ZetaReader(out_path, mesh=single_device_mesh)
    assert zr.slab_io is not None
    zr.close()
    assert zr.slab_io is None
    # double-close is a no-op
    zr.close()


def test_zeta_reader_slab_io_property(tmp_path, single_device_mesh):
    """The exposed SlabIO handle is the one we can hand to legacy
    callers still on the r-space contract."""
    out_path = _build_zeta_h5(tmp_path, n_q_disk=4)
    with ZetaReader(out_path, mesh=single_device_mesh) as zr:
        # SlabIO type — we don't import it here, just check the
        # attribute exists and has a read_slab method.
        assert hasattr(zr.slab_io, 'read_slab')

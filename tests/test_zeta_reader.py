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
from file_io.zeta_loader import ZetaLoader as ZetaReader

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
    # merged ZetaLoader contract: use-after-close raises instead of
    # returning None (catches stale-handle bugs at the seam).
    with pytest.raises(RuntimeError):
        _ = zr.slab_io
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


# ---------------------------------------------------------------------------
# μ-axis pad-past-extent contract — locks in the per-rank clamping that
# both backends must honour when the caller pads ``n_rmu`` to a
# mesh-product but the on-disk dataset stops at the logical extent.
# ---------------------------------------------------------------------------

def _build_zeta_h5_gflat(tmp_path, n_q_disk: int, n_rmu: int, n_G_sph: int,
                          fill_marker: complex = (1.0 + 2.0j)):
    """Build a synthetic G_flat ``zeta_q_G`` file with a non-trivial fill
    so we can distinguish file data from zero-pad in the reader output."""
    wfn_path = str(tmp_path / "WFN_gflat.h5")
    out_path = str(tmp_path / "zeta_q_gflat.h5")
    _make_fake_wfn(wfn_path)
    copy_mf_header(wfn_path, out_path, dst_mode='w')

    # G-flat layout requires gvec_components / ngk_per_q / zeta_cutoff_ry.
    gvec_components = np.zeros((n_q_disk, 3, n_G_sph), dtype=np.int32)
    ngk_per_q = np.full((n_q_disk,), n_G_sph, dtype=np.int32)
    hdr = IsdfHeader.build(
        r_mu_fft_idx=np.arange(3 * n_rmu, dtype=np.int32).reshape(n_rmu, 3) % 8,
        fft_grid=(8, 8, 8),
        density='scalar',
        vertex_mu_L=0,
        zeta_layout='G_flat',
        gvec_components=gvec_components,
        ngk_per_q=ngk_per_q,
        zeta_cutoff_ry=10.0,
    )
    write_isdf_header(out_path, hdr, mode='a')

    # Encode each (q, mu, g) cell with a unique value so we can verify
    # exact positional read-back AND verify zero-pad on the trailing
    # μ slots that the writer never touched.
    payload = np.zeros((n_q_disk, n_rmu, n_G_sph), dtype=np.complex128)
    for q in range(n_q_disk):
        for m in range(n_rmu):
            for g in range(n_G_sph):
                payload[q, m, g] = fill_marker + complex(q, m * 100 + g)
    with h5py.File(out_path, 'a') as f:
        f.create_dataset('zeta_q_G', data=payload)
    return out_path, payload


def test_read_zeta_G_slab_zero_pads_trailing_mu(tmp_path, single_device_mesh):
    """When ``mu_count > valid_mu`` the reader must return the on-disk
    prefix in the first ``valid_mu`` μ slots and exact zeros in the
    pad — same contract the C++ FFI handler honours per-rank under a
    multi-device mesh.

    This is the unit-level analogue of the production scenario at
    CrI3 6×6 30Ry bispinor with ``n_rmu_logical=300``,
    ``n_rmu_padded=304`` on a 16-rank mesh.  Here we shrink to
    ``n_rmu_logical=3, n_rmu_padded=4`` on a 1×1 mesh — sufficient to
    lock the (caller-level) pad-past-extent contract; the per-rank
    clipping is exercised under SLURM at full scale.
    """
    n_q_disk, n_rmu_logical, n_G_sph = 2, 3, 4
    n_rmu_padded = 4  # round-up to mesh product (×1 here, ×16 in prod)

    out_path, payload = _build_zeta_h5_gflat(
        tmp_path, n_q_disk=n_q_disk, n_rmu=n_rmu_logical, n_G_sph=n_G_sph)

    with ZetaReader(out_path, mesh=single_device_mesh) as zr:
        assert zr.zeta_layout == 'G_flat'
        assert zr.n_rmu == n_rmu_logical
        assert zr.n_rmu_disk == n_rmu_logical
        assert zr.n_G_sph_disk == n_G_sph

        zeta_g = zr.read_zeta_G_slab(
            q_offset=0, q_count=n_q_disk,
            mu_offset=0, mu_count=n_rmu_padded,
            qvec_batch_frac=jax.numpy.zeros((n_q_disk, 3),
                                            dtype=jax.numpy.float64),
            sphere_idx=None,
            valid_mu=n_rmu_logical,
        )
        host = np.asarray(zeta_g)

    # Physical shape is the padded extent.
    assert host.shape == (n_q_disk, n_rmu_padded, n_G_sph), host.shape
    # First ``n_rmu_logical`` μ slots match the on-disk payload exactly.
    np.testing.assert_array_equal(host[:, :n_rmu_logical, :], payload)
    # The pad slot(s) at the tail are exact zeros — no garbage from
    # uninitialised host buffer; no read past EOF.
    np.testing.assert_array_equal(
        host[:, n_rmu_logical:, :],
        np.zeros((n_q_disk, n_rmu_padded - n_rmu_logical, n_G_sph),
                 dtype=np.complex128))


def test_read_zeta_G_slab_pad_smaller_than_one_per_rank(
        tmp_path, single_device_mesh):
    """Edge case: when ``n_rmu_logical < n_rmu_padded`` AND
    ``n_rmu_padded // world == 1``, ranks past rank 0 must produce
    zeros (their ``local_start ≥ valid_mu``).  Captures the audit's
    "world_size ∤ n_rmu_logical" trigger explicitly at the
    smallest-meaningful scale."""
    n_q_disk, n_rmu_logical, n_G_sph = 1, 1, 2
    n_rmu_padded = 2

    out_path, payload = _build_zeta_h5_gflat(
        tmp_path, n_q_disk=n_q_disk, n_rmu=n_rmu_logical, n_G_sph=n_G_sph)

    with ZetaReader(out_path, mesh=single_device_mesh) as zr:
        zeta_g = zr.read_zeta_G_slab(
            q_offset=0, q_count=n_q_disk,
            mu_offset=0, mu_count=n_rmu_padded,
            qvec_batch_frac=jax.numpy.zeros((n_q_disk, 3),
                                            dtype=jax.numpy.float64),
            sphere_idx=None,
            valid_mu=n_rmu_logical,
        )
        host = np.asarray(zeta_g)

    assert host.shape == (n_q_disk, n_rmu_padded, n_G_sph), host.shape
    np.testing.assert_array_equal(host[:, :n_rmu_logical, :], payload)
    np.testing.assert_array_equal(
        host[:, n_rmu_logical:, :],
        np.zeros((n_q_disk, n_rmu_padded - n_rmu_logical, n_G_sph),
                 dtype=np.complex128))

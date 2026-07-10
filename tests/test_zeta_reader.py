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


# ===========================================================================
#  r-space ZetaLoader surface (merged from test_zeta_loader.py, 2026-07-09)
# ===========================================================================

# extra imports for the merged r-space ZetaLoader cases
from jax.sharding import PartitionSpec as P  # noqa: F401
from file_io.isdf_header import (
    mark_zeta_done, read_isdf_header,
)
from file_io.zeta_loader import ZetaLoader


def _build_zeta_h5_rspace(tmp_path, n_q_disk: int, *, n_rtot: int = 8,
                   n_rmu: int = 4, density: str = "scalar",
                   vertex_mu_L: int = 0, zeta_is_done: bool = False,
                   fill: complex = 0.0):
    wfn_path = str(tmp_path / "WFN.h5")
    out_path = str(tmp_path / "zeta_q.h5")
    _make_fake_wfn(wfn_path)
    copy_mf_header(wfn_path, out_path, dst_mode='w')

    hdr = IsdfHeader.build(
        r_mu_fft_idx=np.arange(3 * n_rmu, dtype=np.int32).reshape(n_rmu, 3) % 8,
        fft_grid=(8, 8, 8),
        density=density, vertex_mu_L=vertex_mu_L,
        zeta_is_done=zeta_is_done,
    )
    write_isdf_header(out_path, hdr, mode='a')

    with h5py.File(out_path, 'a') as f:
        # Distinct values per (q, r, μ) so the loader's slicing is testable.
        data = (np.arange(n_q_disk * n_rtot * n_rmu)
                .reshape(n_q_disk, n_rtot, n_rmu).astype(np.complex128))
        if fill:
            data = data + fill
        f.create_dataset('zeta_q', data=data)
    return out_path


def test_mark_zeta_done_flips_flag(tmp_path):
    out = _build_zeta_h5_rspace(tmp_path, n_q_disk=4)
    assert read_isdf_header(out).zeta_is_done is False
    mark_zeta_done(out)
    assert read_isdf_header(out).zeta_is_done is True
    # Idempotent
    mark_zeta_done(out)
    assert read_isdf_header(out).zeta_is_done is True


def test_legacy_files_without_flag_read_as_done(tmp_path):
    """A pre-flag zeta_q.h5 (no zeta_is_done dataset) reads as True."""
    out = _build_zeta_h5_rspace(tmp_path, n_q_disk=4)
    # Remove the dataset to simulate a legacy file.
    with h5py.File(out, 'a') as f:
        del f['isdf_header/zeta_is_done']
    hdr = read_isdf_header(out)
    assert hdr.zeta_is_done is True


def test_zeta_layout_round_trip(tmp_path):
    """``zeta_layout`` round-trips for both 'r_space' (default) and 'G_flat'."""
    out_r = _build_zeta_h5_rspace(tmp_path, n_q_disk=2)
    hdr_r = read_isdf_header(out_r)
    assert hdr_r.zeta_layout == 'r_space'

    # Build a synthetic G-flat header to confirm round-trip.
    out_g = tmp_path / 'zeta_q_g.h5'
    from file_io.mf_header import copy_mf_header
    wfn_path = str(tmp_path / 'WFN.h5')
    _make_fake_wfn(wfn_path)
    copy_mf_header(wfn_path, str(out_g), dst_mode='w')
    g_hdr = IsdfHeader.build(
        r_mu_fft_idx=np.zeros((4, 3), dtype=np.int32),
        fft_grid=(8, 8, 8),
        density='scalar', vertex_mu_L=0,
        zeta_is_done=True, zeta_layout='G_flat',
        # G-flat layout now carries the per-q sphere metadata
        # (WFN.h5 ``coeffs`` style); supply trivial values for the
        # round-trip test.
        gvec_components=np.zeros((1, 3, 4), dtype=np.int32),
        ngk_per_q=np.array([4], dtype=np.int32),
        zeta_cutoff_ry=30.0,
    )
    write_isdf_header(str(out_g), g_hdr, mode='a')
    hdr_g = read_isdf_header(str(out_g))
    assert hdr_g.zeta_layout == 'G_flat'


def test_zeta_layout_legacy_default(tmp_path):
    """Files without ``zeta_layout`` read as 'r_space'."""
    out = _build_zeta_h5_rspace(tmp_path, n_q_disk=2)
    with h5py.File(out, 'a') as f:
        del f['isdf_header/zeta_layout']
    hdr = read_isdf_header(out)
    assert hdr.zeta_layout == 'r_space'


def test_zeta_layout_rejects_garbage(tmp_path):
    """build() rejects non-{'r_space','G_flat'} values."""
    with pytest.raises(ValueError, match="zeta_layout must be"):
        IsdfHeader.build(
            r_mu_fft_idx=np.zeros((1, 3), dtype=np.int32),
            fft_grid=(4, 4, 4),
            density='scalar', vertex_mu_L=0,
            zeta_layout='bogus',
        )


# ---------------------------------------------------------------------------
# .load surface
# ---------------------------------------------------------------------------

def test_load_q_ibz_returns_all_disk_rows(tmp_path, single_device_mesh):
    out = _build_zeta_h5_rspace(tmp_path, n_q_disk=6, n_rmu=4)
    with ZetaLoader(out, mesh=single_device_mesh) as ld:
        z = np.asarray(ld.load(q='ibz'))
        assert z.shape == (6, 8, 4)


def test_load_q_index_list_subset(tmp_path, single_device_mesh):
    out = _build_zeta_h5_rspace(tmp_path, n_q_disk=6, n_rmu=4)
    with ZetaLoader(out, mesh=single_device_mesh) as ld:
        z = np.asarray(ld.load(q=[2, 3, 4]))
        assert z.shape == (3, 8, 4)
        # Should match the q=2,3,4 slice of the on-disk pattern.
        with h5py.File(out, 'r') as f:
            ref = f['zeta_q'][2:5]
        np.testing.assert_array_equal(z, ref)


def test_load_mu_range(tmp_path, single_device_mesh):
    out = _build_zeta_h5_rspace(tmp_path, n_q_disk=4, n_rmu=8)
    with ZetaLoader(out, mesh=single_device_mesh) as ld:
        z = np.asarray(ld.load(q='ibz', mu=(2, 6)))
        assert z.shape == (4, 8, 4)
        with h5py.File(out, 'r') as f:
            ref = f['zeta_q'][:, :, 2:6]
        np.testing.assert_array_equal(z, ref)


def test_load_q_full_bz_on_full_disk_returns_all(tmp_path, single_device_mesh):
    """When the on-disk layout IS full-BZ, q='full_bz' == q='ibz'."""
    out = _build_zeta_h5_rspace(tmp_path, n_q_disk=24)   # full = 2*3*4
    with ZetaLoader(out, mesh=single_device_mesh) as ld:
        assert ld.q_layout == 'full_bz'
        z = np.asarray(ld.load(q='full_bz'))
        assert z.shape == (24, 8, 4)


def test_load_q_full_bz_unfold_path_exists(tmp_path, single_device_mesh):
    """The IBZ → full-BZ unfold is wired and dispatches.

    The synthetic ``_make_fake_wfn`` writes random ``mtrx`` entries
    that are typically singular, so ``compute_rgrid_sym_perm`` raises
    ``LinAlgError`` / ``RuntimeError`` for fake files.  We accept any
    of: clean unfold (would require valid sym group), the singular
    matrix error from the fake WFN, or the TR-mapping
    ``NotImplementedError`` — all confirm the unfold path was
    reached.  End-to-end correctness comes from the MoS2 3×3 smoke
    against real WFN symmetries.
    """
    out = _build_zeta_h5_rspace(tmp_path, n_q_disk=1)
    with ZetaLoader(out, mesh=single_device_mesh) as ld:
        try:
            z = np.asarray(ld.load(q='full_bz'))
            # If we got here the syms were valid; identity ⇒ all rows equal.
            with h5py.File(out, 'r') as f:
                ibz = f['zeta_q'][0]
            for q in range(z.shape[0]):
                np.testing.assert_array_equal(z[q], ibz)
        except NotImplementedError:
            pass     # TR-mapping not supported — expected on this fake
        except np.linalg.LinAlgError:
            pass     # Random fake mtrx is singular — expected; smoke covers real case
        except RuntimeError as e:
            if "compute_rgrid_sym_perm" not in str(e):
                raise


def test_load_layout_g_flat_requires_qvec_and_sphere(tmp_path, single_device_mesh):
    out = _build_zeta_h5_rspace(tmp_path, n_q_disk=4)
    with ZetaLoader(out, mesh=single_device_mesh) as ld:
        with pytest.raises(ValueError, match=r"layout='G_flat'\) requires"):
            ld.load(q='ibz', layout='G_flat')

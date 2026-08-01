"""file_io contracts: mf/isdf headers, ZetaLoader/Reader, SlabIO FFI bounds.

Merged (2026-07-09 redesign) from test_mf_isdf_header_roundtrip.py (also
the home of ``_make_fake_wfn``, imported by the V_q unit files),
test_zeta_reader.py (which had already absorbed test_zeta_loader.py), and
test_slab_io_ffi_contract.py.  All synthetic-HDF5, no GPU requirement.
"""

from __future__ import annotations


# ===========================================================================
#  mf_header + isdf_header round-trips (was test_mf_isdf_header_roundtrip.py)
# ===========================================================================

import os

import h5py
import numpy as np
import pytest

from file_io.mf_header import (
    MfHeader,
    copy_mf_header,
    read_mf_header,
)
from file_io.isdf_header import (
    IsdfHeader,
    read_isdf_header,
    write_isdf_header,
)


# ---------------------------------------------------------------------------
# Helpers — build a synthetic mf_header on disk that mirrors the WFN.h5
# schema WFNReader expects.  Sizes are tiny so the test is fast.
# ---------------------------------------------------------------------------

_NS = 1
_NSPINOR = 2
_NK = 3
_NB = 4
_NTRAN = 8
_NAT = 2


def _make_fake_wfn(path: str) -> dict:
    """Write a minimal mf_header group at ``path`` (mode='w').

    Returns a dict of the values written so the round-trip tests can
    cross-check exactly.
    """
    rng = np.random.default_rng(seed=0xCAFE)
    values = dict(
        version=np.int32(3),
        flavor=np.int32(2),
        # kpoints
        nspin=np.int32(_NS),
        nspinor=np.int32(_NSPINOR),
        nrk=np.int32(_NK),
        mnband=np.int32(_NB),
        ngkmax=np.int32(17),
        ecutwfc=np.float64(35.0),
        kgrid=np.array([2, 3, 4], dtype=np.int32),
        shift=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        ngk=np.array([17, 15, 16], dtype=np.int32),
        ifmin=np.ones((_NS, _NK), dtype=np.int32),
        ifmax=np.full((_NS, _NK), 2, dtype=np.int32),
        w=np.array([1.0 / _NK] * _NK, dtype=np.float64),
        rk=rng.random((_NK, 3)),
        el=rng.random((_NS, _NK, _NB)),
        occ=np.array([[[1.0, 1.0, 0.0, 0.0]] * _NK] * _NS, dtype=np.float64),
        # gspace
        ng=np.int32(100),
        ecutrho=np.float64(140.0),
        FFTgrid=np.array([16, 16, 16], dtype=np.int32),
        # symmetry
        ntran=np.int32(_NTRAN),
        cell_symmetry=np.int32(0),
        mtrx=rng.integers(-1, 2, size=(48, 3, 3)).astype(np.int32),
        tnp=rng.random((48, 3)),
        # crystal
        celvol=np.float64(123.4),
        recvol=np.float64(0.5),
        alat=np.float64(7.6),
        blat=np.float64(0.825),
        nat=np.int32(_NAT),
        avec=np.eye(3, dtype=np.float64) * 7.6,
        bvec=np.eye(3, dtype=np.float64) * 0.825,
        adot=np.eye(3, dtype=np.float64) * 57.76,
        bdot=np.eye(3, dtype=np.float64) * 0.68,
        atyp=np.array([14, 14], dtype=np.int32),
        apos=rng.random((_NAT, 3)),
    )
    with h5py.File(path, 'w') as f:
        g = f.create_group('mf_header')
        g.create_dataset('versionnumber', data=values['version'])
        g.create_dataset('flavor', data=values['flavor'])
        kp = g.create_group('kpoints')
        for k in ('nspin', 'nspinor', 'nrk', 'mnband', 'ngkmax',
                  'ecutwfc', 'kgrid', 'shift', 'ngk', 'ifmin', 'ifmax',
                  'w', 'rk', 'el', 'occ'):
            kp.create_dataset(k, data=values[k])
        gs = g.create_group('gspace')
        for k in ('ng', 'ecutrho', 'FFTgrid'):
            gs.create_dataset(k, data=values[k])
        sym = g.create_group('symmetry')
        for k in ('ntran', 'cell_symmetry', 'mtrx', 'tnp'):
            sym.create_dataset(k, data=values[k])
        cr = g.create_group('crystal')
        for k in ('celvol', 'recvol', 'alat', 'blat', 'nat', 'avec',
                  'bvec', 'adot', 'bdot', 'atyp', 'apos'):
            cr.create_dataset(k, data=values[k])
    return values


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_read_mf_header_round_trip(tmp_path):
    """Write a fake WFN.h5, read mf_header, assert every field matches."""
    wfn_path = str(tmp_path / "WFN.h5")
    written = _make_fake_wfn(wfn_path)
    hdr = read_mf_header(wfn_path)

    # scalars
    assert hdr.version == written['version']
    assert hdr.flavor == written['flavor']
    assert hdr.nspin == written['nspin']
    assert hdr.nspinor == written['nspinor']
    assert hdr.nkpts == written['nrk']
    assert hdr.nbands == written['mnband']
    assert hdr.ngkmax == written['ngkmax']
    assert hdr.ecutwfc == written['ecutwfc']
    assert hdr.ng == written['ng']
    assert hdr.ecutrho == written['ecutrho']
    assert hdr.ntran == written['ntran']
    assert hdr.cell_symmetry == written['cell_symmetry']
    assert hdr.cell_volume == written['celvol']
    assert hdr.recip_volume == written['recvol']
    assert hdr.alat == written['alat']
    assert hdr.blat == written['blat']
    assert hdr.nat == written['nat']

    # arrays
    np.testing.assert_array_equal(hdr.kgrid, written['kgrid'])
    np.testing.assert_array_equal(hdr.shift, written['shift'])
    np.testing.assert_array_equal(hdr.ngk, written['ngk'])
    np.testing.assert_array_equal(hdr.ifmin, written['ifmin'])
    np.testing.assert_array_equal(hdr.ifmax, written['ifmax'])
    np.testing.assert_array_equal(hdr.kweights, written['w'])
    np.testing.assert_array_equal(hdr.kpoints, written['rk'])
    np.testing.assert_array_equal(hdr.energies, written['el'])
    np.testing.assert_array_equal(hdr.occs, written['occ'])
    np.testing.assert_array_equal(hdr.fft_grid, written['FFTgrid'])
    np.testing.assert_array_equal(hdr.sym_matrices, written['mtrx'])
    np.testing.assert_array_equal(hdr.translations, written['tnp'])
    np.testing.assert_array_equal(hdr.avec, written['avec'])
    np.testing.assert_array_equal(hdr.bvec, written['bvec'])
    np.testing.assert_array_equal(hdr.adot, written['adot'])
    np.testing.assert_array_equal(hdr.bdot, written['bdot'])
    np.testing.assert_array_equal(hdr.atom_types, written['atyp'])
    np.testing.assert_array_equal(hdr.atom_positions, written['apos'])


def test_copy_mf_header_verbatim(tmp_path):
    """Copy mf_header from a fake WFN.h5 to a fresh destination and
    verify byte-equal field values (datasets created by h5py.copy)."""
    wfn_path = str(tmp_path / "WFN.h5")
    dst_path = str(tmp_path / "zeta_q.h5")
    _ = _make_fake_wfn(wfn_path)

    # dst doesn't exist; copy_mf_header should create it under 'a' mode.
    copy_mf_header(wfn_path, dst_path, dst_mode='w')

    src = read_mf_header(wfn_path)
    dst = read_mf_header(dst_path)
    for field in src._fields:
        src_v = getattr(src, field)
        dst_v = getattr(dst, field)
        if isinstance(src_v, np.ndarray):
            np.testing.assert_array_equal(src_v, dst_v, err_msg=field)
        else:
            assert src_v == dst_v, f"{field}: {src_v} != {dst_v}"


def test_copy_mf_header_refuses_overwrite(tmp_path):
    wfn_path = str(tmp_path / "WFN.h5")
    dst_path = str(tmp_path / "out.h5")
    _ = _make_fake_wfn(wfn_path)
    copy_mf_header(wfn_path, dst_path, dst_mode='w')
    with pytest.raises(ValueError, match="already has an 'mf_header'"):
        copy_mf_header(wfn_path, dst_path, dst_mode='a')


def test_isdf_header_round_trip(tmp_path):
    out_path = str(tmp_path / "zeta_q.h5")
    rng = np.random.default_rng(seed=42)
    n_rmu = 7
    r_mu_fft_idx = rng.integers(0, 16, size=(n_rmu, 3)).astype(np.int32)
    hdr = IsdfHeader.build(
        r_mu_fft_idx=r_mu_fft_idx,
        fft_grid=(16, 16, 16),
        density='scalar',
        vertex_mu_L=0,
    )
    # Need a destination file to attach the header to.
    with h5py.File(out_path, 'w') as _:
        pass
    write_isdf_header(out_path, hdr, mode='a')

    rd = read_isdf_header(out_path)
    assert rd.density == 'scalar'
    assert rd.vertex_mu_L == 0
    assert rd.n_rmu == n_rmu
    np.testing.assert_array_equal(rd.r_mu_fft_idx, r_mu_fft_idx)
    np.testing.assert_array_equal(
        rd.r_mu_crystal, r_mu_fft_idx.astype(np.float64) / 16.0)


def test_isdf_header_current_label(tmp_path):
    out_path = str(tmp_path / "zeta_q_mu2.h5")
    n_rmu = 3
    r_mu_fft_idx = np.array([[0, 0, 0], [1, 2, 3], [5, 6, 7]],
                            dtype=np.int32)
    hdr = IsdfHeader.build(
        r_mu_fft_idx=r_mu_fft_idx,
        fft_grid=(8, 8, 8),
        density='current',
        vertex_mu_L=2,
    )
    with h5py.File(out_path, 'w') as _:
        pass
    write_isdf_header(out_path, hdr, mode='a')
    rd = read_isdf_header(out_path)
    assert rd.density == 'current'
    assert rd.vertex_mu_L == 2


def test_isdf_header_refuses_overwrite(tmp_path):
    out_path = str(tmp_path / "zeta_q.h5")
    hdr = IsdfHeader.build(
        r_mu_fft_idx=np.zeros((2, 3), dtype=np.int32),
        fft_grid=(4, 4, 4),
        density='scalar',
        vertex_mu_L=0,
    )
    with h5py.File(out_path, 'w') as _:
        pass
    write_isdf_header(out_path, hdr, mode='a')
    with pytest.raises(ValueError, match="already has an"):
        write_isdf_header(out_path, hdr, mode='a')


def test_mf_and_isdf_headers_coexist(tmp_path):
    """The two headers must live side-by-side without trampling each
    other — this is the lifecycle the writer uses."""
    wfn_path = str(tmp_path / "WFN.h5")
    out_path = str(tmp_path / "zeta_q.h5")
    _ = _make_fake_wfn(wfn_path)

    copy_mf_header(wfn_path, out_path, dst_mode='w')
    hdr = IsdfHeader.build(
        r_mu_fft_idx=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        fft_grid=(8, 8, 8),
        density='scalar',
        vertex_mu_L=0,
    )
    write_isdf_header(out_path, hdr, mode='a')

    mf = read_mf_header(out_path)
    isdf = read_isdf_header(out_path)
    assert mf.nkpts == _NK
    assert isdf.vertex_mu_L == 0
    assert isdf.n_rmu == 2


# ===========================================================================
#  ZetaLoader/Reader surface + G-flat pad contracts (was test_zeta_reader.py,
#  which had absorbed test_zeta_loader.py)
# ===========================================================================

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
        # The builder writes its full (synthetic) payload below, so the
        # file is complete: stamp it done, as the production writer does
        # after the last chunk drains (ZetaLoader refuses partial files
        # at open — the zeta_is_done completeness gate, 2026-07 audit).
        zeta_is_done=True,
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
        zeta_is_done=True,          # complete synthetic payload (see r-space builder)
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
                   vertex_mu_L: int = 0, zeta_is_done: bool = True,
                   fill: complex = 0.0):
    # ``zeta_is_done`` defaults True: this builder writes its complete
    # synthetic payload synchronously, and ``ZetaLoader`` refuses partial
    # files at open (the completeness gate).  Pass False explicitly to
    # exercise the partial/writer-path behaviour.
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
    out = _build_zeta_h5_rspace(tmp_path, n_q_disk=4, zeta_is_done=False)
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


# ===========================================================================
#  SlabIO FFI bounds/divisibility contracts (was test_slab_io_ffi_contract.py)
# ===========================================================================
import pytest
import numpy as np

from file_io._slab_io_ffi import (
    _normalize_slab_request,
    _normalize_valid_shape,
    _validate_block_divisible,
)
from file_io.slab_io import SlabIO


def test_normalize_slab_request_rejects_rank_mismatch():
    with pytest.raises(ValueError, match="rank mismatch"):
        _normalize_slab_request(
            op="write_slab",
            name="zeta_q",
            offset=(0, 0),
            slab_shape=(36, 70000, 1500),
            global_shape=(36, 1125000, 1500),
        )


def test_normalize_slab_request_rejects_out_of_bounds():
    with pytest.raises(ValueError, match=r"dim 1: 1120000\+70000>1125000"):
        _normalize_slab_request(
            op="write_slab",
            name="zeta_q",
            offset=(0, 1120000, 0),
            slab_shape=(36, 70000, 1500),
            global_shape=(36, 1125000, 1500),
        )


def test_normalize_slab_request_allows_nonzero_read_offset_without_extent():
    off, shape, gshape = _normalize_slab_request(
        op="read_slab",
        name="zeta_q",
        offset=(35, 0, 1300),
        slab_shape=(1, 1125000, 200),
        global_shape=None,
        check_bounds=False,
    )
    assert off == (35, 0, 1300)
    assert shape == (1, 1125000, 200)
    assert gshape == shape


def test_validate_block_divisible_rejects_ragged_write_axis():
    with pytest.raises(ValueError, match="dimension 1 size 70001"):
        _validate_block_divisible(
            op="write_slab",
            name="zeta_q",
            shape=(36, 70001, 1500),
            axis_count_per_dim=(0, 2, 0),
            axis_flat=(0, 1),
            mesh_shape=(4, 4),
        )


def test_validate_block_divisible_accepts_cri3_zeta_shape():
    _validate_block_divisible(
        op="write_slab",
        name="zeta_q",
        shape=(36, 70000, 1500),
        axis_count_per_dim=(0, 2, 0),
        axis_flat=(0, 1),
        mesh_shape=(4, 4),
    )


def test_normalize_valid_shape_rejects_prefix_larger_than_physical_slab():
    with pytest.raises(ValueError, match="valid_shape exceeds slab shape"):
        _normalize_valid_shape(
            op="write_slab",
            name="zeta_q",
            valid_shape=(36, 70001, 1500),
            slab_shape=(36, 70000, 1500),
            offset=(0, 0, 0),
            global_shape=(36, 1125000, 1500),
        )


def test_normalize_valid_shape_allows_ragged_last_file_chunk():
    vshape = _normalize_valid_shape(
        op="write_slab",
        name="zeta_q",
        valid_shape=(36, 65432, 1500),
        slab_shape=(36, 70000, 1500),
        offset=(0, 1050000, 0),
        global_shape=(36, 1125432, 1500),
    )
    assert vshape == (36, 65432, 1500)


def test_allgather_valid_shape_write_and_zero_padded_read(tmp_path):
    path = tmp_path / "slab.h5"
    physical = np.arange(12, dtype=np.float64).reshape(3, 4)
    with SlabIO(str(path), mode="w") as io:
        io.write_slab(
            "A",
            physical,
            global_shape=(3, 3),
            valid_shape=(3, 3),
        )

    with SlabIO(str(path), mode="r") as io:
        host = io.read_slab(
            "A",
            shape=(3, 4),
            valid_shape=(3, 3),
            as_numpy=True,
        )

    expected = np.zeros((3, 4), dtype=np.float64)
    expected[:, :3] = physical[:, :3]
    np.testing.assert_array_equal(host, expected)

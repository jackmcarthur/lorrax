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
from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from zeta_loader import ZetaLoader as ZetaReader  # noqa: E402

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
from zeta_loader import ZetaLoader


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
# .load surface — G-FLAT ONLY since 2026-08-07
#
# These cells used to drive ``_build_zeta_h5_rspace`` files, because
# ``load``'s default layout was ``'r_space'``.  That data surface is gone
# (zeta_loader design D1/D3: no writer in the tree emits r_space), so the
# same q/μ windowing coverage now runs on the G-flat builder and the
# r-space cases become REFUSAL cells further down.  What is deliberately
# NOT here any more is ``test_load_q_full_bz_unfold_path_exists``: the
# ζ(r) IBZ→full-BZ unfold it reached is deleted, and the cell asserted
# only that the path was *reached* (it accepted a clean answer OR
# LinAlgError OR NotImplementedError OR a fft_grid_pullback_perm
# RuntimeError, because the fake WFN's mtrx is singular).  Its successor
# is the refusal cell, which asserts something.
# ---------------------------------------------------------------------------

def test_load_q_ibz_returns_all_disk_rows(tmp_path, single_device_mesh):
    out, _payload = _build_zeta_h5_gflat(
        tmp_path, n_q_disk=6, n_rmu=4, n_G_sph=5)
    with ZetaLoader(out, mesh=single_device_mesh) as ld:
        z = np.asarray(ld.load(q='ibz'))
        assert z.shape == (6, 4, 5)


def test_load_q_index_list_subset(tmp_path, single_device_mesh):
    out, _payload = _build_zeta_h5_gflat(
        tmp_path, n_q_disk=6, n_rmu=4, n_G_sph=5)
    with ZetaLoader(out, mesh=single_device_mesh) as ld:
        z = np.asarray(ld.load(q=[2, 3, 4]))
        assert z.shape == (3, 4, 5)
        # Should match the q=2,3,4 slice of the on-disk pattern.
        with h5py.File(out, 'r') as f:
            ref = f['zeta_q_G'][2:5]
        np.testing.assert_array_equal(z, ref)


def test_load_mu_range(tmp_path, single_device_mesh):
    out, _payload = _build_zeta_h5_gflat(
        tmp_path, n_q_disk=4, n_rmu=8, n_G_sph=5)
    with ZetaLoader(out, mesh=single_device_mesh) as ld:
        z = np.asarray(ld.load(q='ibz', mu=(2, 6)))
        assert z.shape == (4, 4, 5)
        with h5py.File(out, 'r') as f:
            ref = f['zeta_q_G'][:, 2:6, :]
        np.testing.assert_array_equal(z, ref)


def test_load_q_full_bz_on_full_disk_returns_all(tmp_path, single_device_mesh):
    """When the on-disk layout IS full-BZ, q='full_bz' == q='ibz'."""
    out, _payload = _build_zeta_h5_gflat(
        tmp_path, n_q_disk=24, n_rmu=4, n_G_sph=5)   # full = 2*3*4
    with ZetaLoader(out, mesh=single_device_mesh) as ld:
        assert ld.q_layout == 'full_bz'
        z = np.asarray(ld.load(q='full_bz'))
        assert z.shape == (24, 4, 5)


def test_load_q_full_bz_on_an_ibz_file_refuses_and_names_the_post_v_q_unfold(
        tmp_path):
    """The successor to ``test_load_q_full_bz_unfold_path_exists``.

    ``mesh=None`` on purpose: the refusal is a LAYOUT/geometry decision
    taken before any transport is touched, so this cell runs on a stack
    with no phdf5 FFI — which is where the deleted cell could not run at
    all.  The message must name the post-V_q route, because that is the
    thing a caller is supposed to do instead.
    """
    out, _payload = _build_zeta_h5_gflat(
        tmp_path, n_q_disk=6, n_rmu=4, n_G_sph=5)     # 6 < 24 ⇒ IBZ on disk
    ld = ZetaLoader(out)
    assert ld.q_layout == 'ibz'
    # HAND-TRACKED THROUGH THE 2026-08-08 RENAME SWEEP — DELIBERATELY THE
    # OLD SPELLING.  The message being matched is raised by ZetaLoader
    # (``services/zeta_loader``), which this branch may not touch, and it
    # names the route as ``common.symmetry_maps.unfold_v_q``.  Rewriting
    # the pattern to ``unfold_isdf_operator`` here would make this cell
    # red against an unchanged producer; rewriting the producer is out of
    # bounds.  The old name still resolves through the compat alias, so
    # the advice the message gives is still correct.  Both move together
    # at the documentation pass, which is when the aliases retire.
    with pytest.raises(NotImplementedError, match=r"unfold_v_q"):
        ld.load(q='full_bz')


# ---------------------------------------------------------------------------
# r-space ζ: the header still reads, the DATA surface refuses
# ---------------------------------------------------------------------------
# The refusal that replaces every deleted r-space cell.  ``mesh=None``
# throughout, deliberately: __init__ must keep working on an r-space file
# (the header surface is layout-independent and several callers want only
# it), and the data methods must refuse BEFORE they reach the transport,
# which is what lets these run on WSL.

def test_r_space_header_surface_still_opens(tmp_path):
    out = _build_zeta_h5_rspace(tmp_path, n_q_disk=6, n_rmu=4)
    ld = ZetaLoader(out)
    assert ld.zeta_layout == 'r_space'
    assert ld.n_q_on_disk == 6
    assert ld.n_rmu_disk == 4
    assert ld.n_rtot_disk == 8
    assert int(np.prod(ld.kgrid)) == 24


@pytest.mark.parametrize("call", [
    lambda ld: ld.load(q='ibz'),
    lambda ld: ld.read_zeta_G_slab(q_offset=0, q_count=1,
                                   mu_offset=0, mu_count=1),
])
def test_r_space_data_reads_refuse_naming_the_removal(tmp_path, call):
    out = _build_zeta_h5_rspace(tmp_path, n_q_disk=6, n_rmu=4)
    ld = ZetaLoader(out)
    with pytest.raises(ValueError, match=r"2026-08-07"):
        call(ld)


def test_the_g_flat_data_reads_do_not_hit_that_refusal(tmp_path):
    """The does-not-cry-wolf half: a G-flat file must get PAST the layout
    gate.  Without this the refusal above would pass with the gate wired
    to ``raise`` unconditionally, and both data methods would be dead."""
    out, _payload = _build_zeta_h5_gflat(
        tmp_path, n_q_disk=2, n_rmu=3, n_G_sph=4)
    ld = ZetaLoader(out)                       # header-only: no transport
    # It refuses for the MESH reason, not the layout one — i.e. the layout
    # gate let it through.
    with pytest.raises(RuntimeError, match=r"HEADER-ONLY"):
        ld.load(q='ibz')


def test_the_deleted_read_arguments_are_gone_from_the_signature(tmp_path):
    """D3, pinned.  ``qvec_batch_frac``/``sphere_idx`` on
    ``read_zeta_G_slab`` and ``layout``/``qvec_frac``/``sphere_idx`` on
    ``load`` were removed on 2026-08-07; a caller that still passes one
    must get a TypeError rather than have it silently ignored."""
    out, _payload = _build_zeta_h5_gflat(
        tmp_path, n_q_disk=2, n_rmu=3, n_G_sph=4)
    ld = ZetaLoader(out)
    with pytest.raises(TypeError):
        ld.load(q='ibz', layout='G_flat')
    with pytest.raises(TypeError):
        ld.read_zeta_G_slab(q_offset=0, q_count=1, mu_offset=0, mu_count=1,
                            sphere_idx=None)
    assert not hasattr(ld, 'read_zeta_r_slab')
    assert not hasattr(ld, 'has_slab_io')


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
from file_io.slab_io import SlabIO, probe_availability


def _require_slabio():
    """Skip, naming the missing capability, when the tile path is absent.

    SlabIO has ONE transport since 2026-08-06.  These three tests used to
    reach the rank-0 allgather tier by omitting ``mesh`` and ``backend``,
    which was legal at exactly one process; with that tier deleted they
    exercise the real collective writer instead -- better coverage, but it
    needs the phdf5 FFI to be present.  A skip that names the probe stage
    is honest; a silent fallback to a second transport is what this
    change removed.
    """
    ok, stage, reason = probe_availability()
    if not ok:
        pytest.skip(f"SlabIO unavailable (probe stage '{stage}'): {reason}")


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
    # INTERNAL post-condition since decisions.md 2026-08-04 — both FFI
    # entry points pad to divisibility before they reach it.
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
            ds_shape=(36, 1125000, 1500),
        )


def test_normalize_valid_shape_allows_ragged_last_file_chunk():
    vshape = _normalize_valid_shape(
        op="write_slab",
        name="zeta_q",
        valid_shape=(36, 65432, 1500),
        slab_shape=(36, 70000, 1500),
        offset=(0, 1050000, 0),
        ds_shape=(36, 1125432, 1500),
    )
    assert vshape == (36, 65432, 1500)


def test_normalize_valid_shape_derives_the_dataset_clip_by_default():
    """decisions.md 2026-08-04: with no ``valid_shape`` the extent is
    ``min(slab, dataset - offset)`` — the arithmetic every call site used
    to do by hand."""
    vshape = _normalize_valid_shape(
        op="write_slab",
        name="zeta_q_G",
        valid_shape=None,
        slab_shape=(36, 70000, 1500),        # padded for mesh divisibility
        offset=(0, 0, 0),
        ds_shape=(36, 69997, 1500),          # LOGICAL extent on disk
    )
    assert vshape == (36, 69997, 1500)


def test_normalize_valid_shape_default_clips_a_sub_slab_at_its_offset():
    vshape = _normalize_valid_shape(
        op="write_slab", name="zeta_q_G", valid_shape=None,
        slab_shape=(4, 8, 3), offset=(0, 6, 0), ds_shape=(4, 10, 3))
    assert vshape == (4, 4, 3)


def test_normalize_valid_shape_default_never_refuses_an_overhang():
    """A slab starting past the end of the dataset derives an EMPTY
    request (a legitimate no-op rendezvous), not a refusal."""
    vshape = _normalize_valid_shape(
        op="read_slab", name="A", valid_shape=None,
        slab_shape=(4, 8), offset=(0, 12), ds_shape=(4, 10))
    assert vshape == (4, 0)


def test_normalize_valid_shape_override_that_overruns_refuses():
    with pytest.raises(ValueError, match="exceeds dataset extent"):
        _normalize_valid_shape(
            op="write_slab", name="A", valid_shape=(4, 8),
            slab_shape=(4, 8), offset=(0, 6), ds_shape=(4, 10))


def test_slabio_implicit_pad_write_and_zero_padded_read(
        tmp_path, single_device_mesh):
    """The padded write/read round-trip with NO padding argument.

    decisions.md 2026-08-04: the caller states the dataset's logical
    shape once and hands over the padded buffer.  Compare against
    ``..._matches_explicit_valid_shape`` below, which pins that this is
    byte-identical to the pre-ruling spelling.
    """
    _require_slabio()
    mesh = single_device_mesh
    path = tmp_path / "slab.h5"
    physical = np.arange(12, dtype=np.float64).reshape(3, 4)
    with SlabIO(str(path), mode="w", mesh=mesh) as io:
        io.create_dataset("A", shape=(3, 3), dtype=np.float64)
        io.write_slab("A", physical)

    with SlabIO(str(path), mode="r", mesh=mesh) as io:
        host = io.read_slab("A", shape=(3, 4), as_numpy=True)

    expected = np.zeros((3, 4), dtype=np.float64)
    expected[:, :3] = physical[:, :3]
    np.testing.assert_array_equal(host, expected)


def test_slabio_implicit_pad_matches_explicit_valid_shape(
        tmp_path, single_device_mesh):
    """BYTE-IDENTITY: dropping ``valid_shape`` must not move a bit.

    The unit-level twin of the ``implicit`` case in
    ``tests/multi_device/phdf5_padded_rank_write.py``.
    """
    _require_slabio()
    mesh = single_device_mesh
    physical = np.arange(12, dtype=np.float64).reshape(3, 4)

    old = tmp_path / "old.h5"
    with SlabIO(str(old), mode="w", mesh=mesh) as io:
        io.create_dataset("A", shape=(3, 3), dtype=np.float64)
        io.write_slab("A", physical, global_shape=(3, 3),
                      valid_shape=(3, 3))
    new = tmp_path / "new.h5"
    with SlabIO(str(new), mode="w", mesh=mesh) as io:
        io.create_dataset("A", shape=(3, 3), dtype=np.float64)
        io.write_slab("A", physical)

    import h5py
    with h5py.File(str(old), "r") as f:
        a = np.asarray(f["A"])
    with h5py.File(str(new), "r") as f:
        b = np.asarray(f["A"])
    assert a.shape == b.shape
    assert a.tobytes() == b.tobytes()

    with SlabIO(str(new), mode="r", mesh=mesh) as io:
        explicit = io.read_slab("A", shape=(3, 4), valid_shape=(3, 3),
                                as_numpy=True)
        implicit = io.read_slab("A", shape=(3, 4), as_numpy=True)
    assert explicit.tobytes() == implicit.tobytes()


def test_create_dataset_refuses_a_geometry_change(
        tmp_path, single_device_mesh):
    """decisions.md 2026-08-04: reuse-if-identical, refuse otherwise.

    The deleted allgather backend used to ``del`` and recreate, silently
    discarding the previous contents.
    """
    _require_slabio()
    mesh = single_device_mesh
    path = tmp_path / "geom.h5"
    with SlabIO(str(path), mode="w", mesh=mesh) as io:
        io.create_dataset("A", shape=(3, 3), dtype=np.float64)
        io.write_slab("A", np.ones((3, 3), dtype=np.float64))

    with SlabIO(str(path), mode="a", mesh=mesh) as io:
        io.create_dataset("A", shape=(3, 3), dtype=np.float64)   # idempotent
        # RuntimeError, not ValueError: the refusal is raised by the C++
        # ``ensure_dataset`` and surfaces through the FFI as
        # ``lorrax_ffi error (1)``.  The deleted allgather backend raised
        # ValueError from Python for the same condition -- so this is the
        # one place the tier deletion changed an exception TYPE, and the
        # message (which names both shapes) is unchanged.
        with pytest.raises((ValueError, RuntimeError),
                           match="already exists"):
            io.create_dataset("A", shape=(4, 4), dtype=np.float64)

    import h5py
    with h5py.File(str(path), "r") as f:
        np.testing.assert_array_equal(np.asarray(f["A"]),
                                      np.ones((3, 3)))


def test_read_slab_without_shape_rounds_up_to_the_mesh(
        tmp_path, single_device_mesh):
    """THE EASY CALL IS THE CORRECT CALL (2026-08-06).

    ``read_slab(name, partition_spec=spec)`` with no ``shape`` returns the
    dataset rounded UP to the mesh-divisible extent, zero-filled past the
    dataset.  Before this, ``shape=None`` meant "the dataset's own shape",
    which REFUSES whenever that shape is not mesh-divisible -- the normal
    case, since N_mu is a physics number and not a multiple of the device
    count.  So every caller computed the rounded-up extent itself, which is
    the nitty-gritty the owner objected to.

    Pinned at 1x1 (where the round-up is the identity) so the CONTRACT is
    covered by the unit suite; ``tests/multi_device/restart_sharded_parity``
    covers the case where the rounding actually moves the extent.
    """
    _require_slabio()
    mesh = single_device_mesh
    path = tmp_path / "autopad.h5"
    a = np.arange(15, dtype=np.float64).reshape(3, 5)
    with SlabIO(str(path), mode="w", mesh=mesh) as io:
        io.create_dataset("A", shape=(3, 5), dtype=np.float64)
        io.write_slab("A", a)

    with SlabIO(str(path), mode="r", mesh=mesh) as io:
        auto = io.read_slab("A", partition_spec=P("x", "y"), as_numpy=True)
        explicit = io.read_slab("A", shape=(3, 5), partition_spec=P("x", "y"),
                                as_numpy=True)
    # Same bytes as the spelling that states the extent by hand.
    assert auto.shape == explicit.shape
    assert auto.tobytes() == explicit.tobytes()
    np.testing.assert_array_equal(auto, a)


@pytest.mark.mesh(4)   # 2x2; the inline guard below fired in EVERY suite run
def test_mesh_divisible_shape_rounds_each_axis_by_its_own_divisor():
    """The rounding rule itself, without a file.

    Each dim is rounded by the product of the mesh sizes of the axes that
    shard it -- NOT by the total device count.  A caller that rounded both
    μ axes by ``device_count`` would get a legal but larger buffer; the
    restart path deliberately does exactly that (``padded_mu_extent``) for
    its own reasons, and the two must not be confused.
    """
    import jax
    from jax.sharding import Mesh
    from file_io.slab_io import mesh_divisible_shape
    devs = jax.devices()
    if len(devs) < 4:
        pytest.skip(f"needs >= 4 devices for a 2x2 mesh; have {len(devs)}")
    mesh = Mesh(np.asarray(devs[:4]).reshape(2, 2), axis_names=("x", "y"))
    assert mesh_divisible_shape((5, 7), mesh, P("x", "y")) == (6, 8)
    assert mesh_divisible_shape((5, 7), mesh, P(None, "y")) == (5, 8)
    assert mesh_divisible_shape((5, 7), mesh, P(None, None)) == (5, 7)
    # Both axes on one dim -> that dim rounds by the product.
    assert mesh_divisible_shape((5,), mesh, P(("x", "y"),)) == (8,)


# ---------------------------------------------------------------------------
# zeta_q.h5 striping — the inode must be created by MPI-IO, not by h5py
# ---------------------------------------------------------------------------
#
# A Lustre stripe layout is fixed at inode CREATE, and the striping hints live
# in the MPI_Info that phdf5's H5Fcreate builds.  So a file whose inode rank-0
# serial h5py created takes the DIRECTORY DEFAULT and ignores the policy
# completely.  That was the state until 2026-08-07: zeta_q.h5 measured
# stripe_count=1 against a resolved policy of 4, while its sibling
# isdf_tensors_*.h5 measured 4.  It costs nothing at 1 rank, which is how it
# survived; at P>1 ROMIO sets cb_nodes = min(stripe_count, nranks), so a
# 1-stripe file pins collective buffering to a single aggregator however many
# ranks are writing.
#
# ``gw/isdf_fitting.fit_zeta_to_h5`` therefore creates the inode with
# SlabIO(mode='w') and appends the headers afterwards with h5py mode='a'.
# The two halves of that claim are tested separately ON PURPOSE: the append
# half runs everywhere, the striping half needs `lfs` and so cannot run in the
# production container.  Merging them would have produced a single test that
# reports "skipped" in the container while silently having checked the append.


def _build_zeta_like_production(tmp_path, mesh):
    """Lay out a file the way fit_zeta_to_h5 does, and return (path, A)."""
    import jax
    import numpy as np
    from file_io.slab_io import SlabIO
    from file_io.mf_header import copy_mf_header

    out_path = str(tmp_path / "zeta_q.h5")
    wfn_path = str(tmp_path / "WFN.h5")
    _make_fake_wfn(wfn_path)

    shape = (2, 8, 4)
    A = jax.device_put(np.arange(int(np.prod(shape)), dtype=np.float64)
                       .reshape(shape).astype(np.complex128))

    # SlabIO creates the inode collectively (H5Fcreate under the striping
    # hints) and closes; THEN rank 0 appends the headers; THEN the payload.
    with SlabIO(out_path, mode='w', mesh=mesh):
        pass
    assert os.path.exists(out_path), \
        "SlabIO(mode='w') did not create the inode"
    copy_mf_header(wfn_path, out_path, dst_mode='a')
    with SlabIO(out_path, mode='a', mesh=mesh) as io:
        io.create_dataset('zeta_q_G', shape=shape, dtype=np.complex128)
        io.write_slab('zeta_q_G', A)
    return out_path, A


def test_zeta_q_headers_append_onto_phdf5_created_inode(
        tmp_path, single_device_mesh):
    """Serial h5py can append the header groups to a PHDF5-created inode.

    This is the property that makes the create-order swap safe, and it is
    what lets the inode be created by MPI-IO (which is what carries the
    striping hints) instead of by h5py.  Runs everywhere.
    """
    import jax
    import numpy as np
    from file_io.slab_io import SlabIO

    out_path, A = _build_zeta_like_production(tmp_path, single_device_mesh)
    expect = np.asarray(jax.device_get(A))

    with h5py.File(out_path, 'r') as f:
        assert 'mf_header' in f, \
            "h5py mode='a' did not append mf_header to the PHDF5 inode"
        np.testing.assert_allclose(f['zeta_q_G'][...], expect)
    with SlabIO(out_path, mode='r', mesh=single_device_mesh) as io:
        got = np.asarray(jax.device_get(io.read_slab('zeta_q_G')))
    np.testing.assert_allclose(got, expect)


def _lfs_stripe_count(path):
    """Lustre stripe count of ``path``, or ``None`` when unknowable.

    ``None`` means "``lfs`` could not answer" — it is NOT a stripe count of
    1, and callers must skip rather than assert on it.  ``lfs`` is absent
    from the production container, which is why the old prestripe helper
    could not use it either.
    """
    import shutil
    import subprocess
    exe = shutil.which("lfs")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "getstripe", "-c", str(path)],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    for tok in out.stdout.split():
        if tok.lstrip("-").isdigit():
            return int(tok)
    return None


def test_zeta_q_inode_gets_striping_policy(tmp_path, single_device_mesh):
    """The zeta_q.h5 inode carries the stripe count the policy resolves to.

    SKIPS without ``lfs`` — i.e. it does NOT run inside the production
    container.  This is the only assertion that catches a regression to
    ``copy_mf_header(..., dst_mode='w')``, so a green container run has not
    tested striping; run it on a Lustre login node to get that coverage.
    """
    import jax
    from file_io._slab_io_ffi import _stripe_policy

    out_path, _ = _build_zeta_like_production(tmp_path, single_device_mesh)

    got_stripes = _lfs_stripe_count(out_path)
    if got_stripes is None:
        pytest.skip("`lfs` unavailable here, so the stripe count cannot be "
                    "read — this test did NOT verify striping")
    want = _stripe_policy(jax.process_count())[0]
    assert got_stripes == want, (
        f"zeta_q.h5 got stripe_count={got_stripes}, policy wants {want}. "
        f"The inode was created by something that does not carry the MPI-IO "
        f"striping hints — check that fit_zeta_to_h5 still creates it with "
        f"SlabIO(mode='w') BEFORE copy_mf_header(dst_mode='a')."
    )


# ===========================================================================
#  create_dataset(attrs=) — the transport WRITES them
# ===========================================================================
# Registered by the isolation-edge landing (`fef002e9`) as a production
# defect and fixed on 2026-08-08.  The transport used to answer
# ``create_dataset(..., attrs=...)`` with a ``warnings.warn`` and no
# write, and it is SlabIO's only transport, so every cluster-written
# ``sigma_mnk.h5`` lost the ``k_storage="ibz"`` stamp that says its k
# axis is a wedge.  ``kin_ion.read_star_map`` is specified so that no
# stamp means the full BZ — the back-compat rule that keeps every
# pre-format file readable — so those files did not fail, they read back
# as full-BZ and every k_irr consumer indexed full-BZ rows into an
# nrk-row array.
#
# The two end-to-end cells for that live in
# ``tests/test_sigma_kirr_extraction.py`` and need the transport, which
# is why the defect survived: on a box with no phdf5 write symbol they
# skip, and every WSL verification for four days was a skip reported as
# a pass.  So the contract is pinned at BOTH altitudes here — the
# transport's own attr writer, which needs nothing but h5py and runs
# everywhere, and the full SlabIO round trip, which needs the library.

def _attr_signature(ds) -> dict:
    """What the FILE holds for each attr — value, shape AND storage type.

    The dtype is in here because it is where the interesting way to get
    this wrong lives: ``ds.attrs["k_storage"] = "ibz"`` stores a
    variable-length UTF-8 string and reads back as ``str``, while the
    same assignment through ``np.asarray`` stores a fixed-length byte
    string that reads back as ``b"ibz"`` — and
    :func:`file_io.kin_ion.read_star_map` calls ``str()`` on what it
    reads, so the second spelling yields the literal ``"b'ibz'"`` and
    refuses the file.  A comparison on values alone would call those two
    files identical.
    """
    out = {}
    for key in ds.attrs:
        val = ds.attrs[key]
        out[key] = (str(ds.attrs.get_id(key).dtype),
                    np.shape(val),
                    val.tolist() if isinstance(val, np.ndarray) else val)
    return out


def _k_irr_stamp():
    """The real stamp, in the writer's own vocabulary — not a toy dict.

    Mixed on purpose: a string (the one with the dtype trap), an int, a
    numpy float and a plain float, because the transport hands all four
    to h5py and the point of the cells below is that it hands them over
    unchanged.
    """
    from file_io.kin_ion import (K_STORAGE_ATTR, K_STORAGE_IBZ,
                                 K_STORAGE_VERSION,
                                 K_STORAGE_VERSION_ATTR)
    return {
        K_STORAGE_ATTR: K_STORAGE_IBZ,
        K_STORAGE_VERSION_ATTR: K_STORAGE_VERSION,
        "n_sym_spatial": 12,
        "nk_full": 9,
        "sigma_star_spread_raw_ev": np.float64(113.08),
        "sigma_star_spread_omega_index": 2.0,
    }


def test_the_attr_writer_writes_what_the_host_path_would_write(tmp_path):
    """THE WSL TWIN, and it needs no transport: same bytes, both ways.

    ``_apply_dataset_attrs`` is the function close() runs inside its
    rank-0 reopen, so calling it on a file h5py made exercises exactly
    the code the cluster runs — without the phdf5 write symbol the
    round-trip cells need.  The reference is a plain
    ``ds.attrs[key] = value``, which is what the host-side writers
    (``gw.kin_ion_io``, ``sigma_output.append_qsgw_datasets_h5``) do, so
    what this asserts is that a stamp written by the transport and the
    same stamp written host-side are the same file, not merely the same
    intention.
    """
    from file_io._slab_io_ffi import _apply_dataset_attrs

    stamp = _k_irr_stamp()
    payload = np.arange(6, dtype=np.complex128).reshape(2, 3)

    transport = tmp_path / "transport.h5"
    with h5py.File(str(transport), "w") as f:
        f.create_dataset("sigma_c_kij_ev", data=payload)
        _apply_dataset_attrs(f, [("sigma_c_kij_ev", stamp)])

    host = tmp_path / "host.h5"
    with h5py.File(str(host), "w") as f:
        ds = f.create_dataset("sigma_c_kij_ev", data=payload)
        for key, val in stamp.items():
            ds.attrs[key] = val

    with h5py.File(str(transport), "r") as f:
        got = _attr_signature(f["sigma_c_kij_ev"])
    with h5py.File(str(host), "r") as f:
        want = _attr_signature(f["sigma_c_kij_ev"])
    assert got == want, (
        "the transport's attr writer and the host path disagree about "
        "what to put in the file for the same stamp")
    assert set(got) == set(stamp), "an attr did not reach the file at all"


def test_the_writers_stamp_survives_its_own_reader(tmp_path):
    """The consequence, end to end at the h5py altitude.

    The cell above pins byte-identity; this one pins that the bytes are
    the ones ``read_star_map`` accepts.  It is the assertion the dtype
    trap would fail: a fixed-length ``k_storage`` reads back as
    ``b"ibz"``, ``str()`` of which is neither ``"ibz"`` nor ``"full"``,
    and the reader refuses.
    """
    from file_io._slab_io_ffi import _apply_dataset_attrs
    from file_io.kin_ion import (IRR_IDX_DATASET, SYM_IDX_DATASET,
                                 read_star_map)

    irr = np.array([0, 1, 1, 0], dtype=np.int32)
    sidx = np.array([0, 0, 3, 5], dtype=np.int32)
    path = tmp_path / "wedge.h5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("sigma_c_kij_ev",
                         data=np.zeros((4, 2, 3, 3), dtype=np.complex128))
        f.create_dataset(IRR_IDX_DATASET, data=irr)
        f.create_dataset(SYM_IDX_DATASET, data=sidx)
        _apply_dataset_attrs(f, [("sigma_c_kij_ev", _k_irr_stamp())])

    star = read_star_map(str(path), "sigma_c_kij_ev", k_axis=1)
    assert star is not None, (
        "the stamp did not survive the writer, so a two-row wedge reads "
        "back as a two-k full-BZ cube")
    np.testing.assert_array_equal(star[0], irr)
    assert star[2] == 12


def test_an_attr_for_a_missing_dataset_refuses(tmp_path):
    """The regression twin for the SHAPE of the defect, not its instance.

    Skipping a stamp whose dataset is not there is the same silent drop
    wearing a plausible excuse, so it raises.  Every caller creates the
    dataset in the same breath as the attrs, so this can only fire on a
    transport bug and should read as one.
    """
    from file_io._slab_io_ffi import _apply_dataset_attrs

    path = tmp_path / "empty.h5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("A", data=np.zeros(3))
        with pytest.raises(KeyError, match="refusing to drop them silently"):
            _apply_dataset_attrs(f, [("B", {"k_storage": "ibz"})])


def test_slabio_create_dataset_attrs_reach_the_file(
        tmp_path, single_device_mesh):
    """THE RED TWIN, on the transport that had the defect.

    This is the cell that was red before 2026-08-08 and could only ever
    be red on a machine with the phdf5 write symbol.  Nothing here is
    about Σ: it is the transport contract, that ``attrs=`` handed to
    ``create_dataset`` is in the file when it closes.
    """
    _require_slabio()
    stamp = _k_irr_stamp()
    path = tmp_path / "attrs.h5"
    with SlabIO(str(path), mode="w", mesh=single_device_mesh) as io:
        io.create_dataset("A", shape=(3, 3), dtype=np.float64, attrs=stamp)
        io.write_slab("A", np.ones((3, 3), dtype=np.float64))

    with h5py.File(str(path), "r") as f:
        got = _attr_signature(f["A"])
        np.testing.assert_array_equal(np.asarray(f["A"]), np.ones((3, 3)))
    missing = sorted(set(stamp) - set(got))
    assert not missing, (
        f"{missing} were handed to create_dataset(attrs=) and are not in "
        f"the file — the transport is discarding the caller's metadata, "
        f"which is how a wedge cube reads back as full-BZ")


def test_slabio_and_h5py_write_attr_identical_files(
        tmp_path, single_device_mesh):
    """The two paths, the same call, the same file.

    ``sigma_mnk.h5`` is written by BOTH — the Σ cubes through SlabIO and
    the QSGW appendix through ``append_qsgw_datasets_h5``'s plain h5py —
    so "the FFI transport writes attrs now" is not enough.  They have to
    be the same attrs, stored the same way, or one file means two things
    depending on which writer got to a dataset.
    """
    _require_slabio()
    stamp = _k_irr_stamp()
    a = np.arange(9, dtype=np.float64).reshape(3, 3)

    through_slabio = tmp_path / "slabio.h5"
    with SlabIO(str(through_slabio), mode="w", mesh=single_device_mesh) as io:
        io.create_dataset("A", shape=(3, 3), dtype=np.float64, attrs=stamp)
        io.write_slab("A", a)

    through_h5py = tmp_path / "h5py.h5"
    with h5py.File(str(through_h5py), "w") as f:
        ds = f.create_dataset("A", data=a)
        for key, val in stamp.items():
            ds.attrs[key] = val

    with h5py.File(str(through_slabio), "r") as f:
        got, got_data = _attr_signature(f["A"]), np.asarray(f["A"])
    with h5py.File(str(through_h5py), "r") as f:
        want, want_data = _attr_signature(f["A"]), np.asarray(f["A"])
    assert got == want
    assert got_data.tobytes() == want_data.tobytes()


def test_slabio_create_dataset_does_not_warn_about_attrs(
        tmp_path, single_device_mesh):
    """The discard's ghost.  A warning here means the drop came back.

    ``attrs`` is data and must be neither dropped nor warned about.
    """
    _require_slabio()
    import warnings

    path = tmp_path / "quiet.h5"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with SlabIO(str(path), mode="w", mesh=single_device_mesh) as io:
            io.create_dataset("A", shape=(3, 3), dtype=np.float64,
                              attrs=_k_irr_stamp())
            io.write_slab("A", np.ones((3, 3), dtype=np.float64))
    # Filtered rather than counted: this opens a real file on a real
    # stack, and turning every unrelated jax/h5py DeprecationWarning into
    # a failure of THIS claim would be a red that means nothing.
    about_attrs = [str(w.message) for w in caught
                   if "attrs" in str(w.message).lower()]
    assert not about_attrs, (
        f"create_dataset(attrs=) warned: {about_attrs}.  A transport that "
        f"drops the caller's data with a warning is the defect; the attrs "
        f"are written now and there is nothing to warn about")

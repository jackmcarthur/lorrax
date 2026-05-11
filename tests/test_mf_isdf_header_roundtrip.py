"""Round-trip tests for file_io.mf_header + file_io.isdf_header.

Builds a synthetic mf_header h5 layout (the BGW WFN.h5 schema, minus
the wfns/* group), copies it into a destination, writes an isdf_header
on top, and reads both back asserting bit-equality.
"""
from __future__ import annotations

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

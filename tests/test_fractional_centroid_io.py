"""Explicit real centroid coordinates survive file I/O and source identity."""
from dataclasses import replace
import hashlib
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from runtime import bootstrap
bootstrap()

from file_io.centroids import (
    load_centroid_basis, load_centroids, read_centroid_coordinate_kind,
)
from file_io.isdf_header import IsdfHeader, bind_isdf_attrs, write_isdf_header
from file_io.restart_bundle import read_isdf_header
from file_io.wfn_basis import (
    CENTROID_TABLE_FINGERPRINT_SCHEME,
    FRACTIONAL_CENTROID_TABLE_FINGERPRINT_SCHEME,
    WavefunctionBasisReceipt, centroid_table_md5,
)
from zeta_loader import probe_zeta_file


def _table(tmp_path, positions, header=None, name='centroids.txt'):
    path = tmp_path / name
    np.savetxt(path, positions, fmt='%.17g', header='' if header is None else header)
    return str(path)


def test_header_is_explicit_and_missing_kind_preserves_legacy_snapping(tmp_path):
    positions = np.array([[-.123456789, .019, 1.213456789], [.13, .27, .31]])
    grid = np.array([8, 8, 8])
    fractional = _table(tmp_path, positions, 'centroid coordinate kind: fractional')
    _, consumed, count = load_centroids(fractional, grid)
    assert read_centroid_coordinate_kind(fractional) == 'fractional'
    assert consumed.dtype == np.float64 and count == 2
    np.testing.assert_array_equal(consumed, positions)
    legacy = _table(tmp_path, positions, name='legacy.txt')
    assert read_centroid_coordinate_kind(legacy) == 'fft_indices'
    np.testing.assert_array_equal(load_centroids(legacy, grid)[1],
                                  np.rint(positions*grid).astype(np.int64) % grid)
    explicit = _table(tmp_path, positions, 'centroid coordinate kind: fft_indices', 'explicit.txt')
    np.testing.assert_array_equal(load_centroids(explicit, grid)[1], load_centroids(legacy, grid)[1])


@pytest.mark.parametrize('header', [
    'centroid coordinate kind: float64', 'centroid coordinate kind:',
    'centroid coordinate kind: fractional\ncentroid coordinate kind: fft_indices',
])
def test_invalid_or_conflicting_coordinate_kind_refuses(tmp_path, header):
    path = _table(tmp_path, [[.1, .2, .3]], header)
    with pytest.raises(ValueError, match='coordinate kind'):
        load_centroids(path, [8, 8, 8])


def test_closure_measures_actual_fractional_positions_before_selection(tmp_path):
    # A translation by 1/3 is closed on this real orbit but not on an 8-grid.
    positions = np.array([[.13+i/3, .2712345, .3134567] for i in range(3)])
    sym = SimpleNamespace(sym_matrices=np.tile(np.eye(3, dtype=int), (3, 1, 1)),
                          translations=2*np.pi*np.array([[i/3, 0, 0] for i in range(3)]))
    path = _table(tmp_path, positions, 'centroid coordinate kind: fractional')
    loaded = load_centroid_basis(path, [8, 8, 8], sym=sym)
    assert loaded.coordinate_kind == 'fractional' and loaded.orbit_closed
    assert loaded.closure.worst_residual < 1e-14
    np.testing.assert_array_equal(loaded.centroid_indices, positions)
    legacy = load_centroid_basis(_table(tmp_path, positions, name='legacy.txt'), [8]*3, sym=sym)
    assert legacy.coordinate_kind == 'fft_indices' and not legacy.orbit_closed
    assert legacy.closure.worst_residual > .01
    subset = load_centroid_basis(path, [8]*3, sym=sym, selection=np.array([2, 0]))
    assert subset.source_n_rmu == 3 and subset.n_rmu == 2 and not subset.orbit_closed
    np.testing.assert_array_equal(subset.centroid_indices, positions[[2, 0]])


def test_fractional_fingerprint_preserves_subgrid_changes_and_byte_order():
    grid = np.array([50, 50, 50])
    first = np.array([[.1234, .271, .319], [.4234, .721, .819]])
    second = first.copy()
    second[0, 0] += 1e-6  # well below half an FFT-grid spacing
    a, b = [np.rint(x*grid).astype(np.int64) for x in (first, second)]
    np.testing.assert_array_equal(a, b)
    historical = hashlib.md5(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest()
    assert centroid_table_md5(a) == historical == centroid_table_md5(b)
    assert centroid_table_md5(first, coordinate_kind='fractional') != centroid_table_md5(second, coordinate_kind='fractional')
    assert centroid_table_md5(first, coordinate_kind='fractional') == centroid_table_md5(first.astype('>f8'), coordinate_kind='fractional')
    assert centroid_table_md5(np.zeros((1, 3), int)) != centroid_table_md5(np.zeros((1, 3)), coordinate_kind='fractional')
    with pytest.raises(ValueError, match='coordinate_kind'):
        centroid_table_md5(first, coordinate_kind='automatic')
    with pytest.raises(ValueError, match='finite'):
        centroid_table_md5([[np.nan, 0, 0]], coordinate_kind='fractional')


def test_receipt_binds_explicit_kind_scheme_and_actual_positions(monkeypatch):
    import common.parallel_transport as source
    monkeypatch.setattr(source, 'wfn_fingerprint', lambda wfn: 'a'*64)
    monkeypatch.setattr(source, 'fingerprint_from_binding', lambda binding, wfn: 'a'*64)
    positions = np.array([[-.13, .27, 1.31], [.63, .77, .81]])
    kwargs = dict(wfn=SimpleNamespace(nbands=4, nspinor=2), role='charge',
                  bispinor=True, bispinor_lift='normalized_rkb', band_interval=(0, 4),
                  fft_grid=(50, 50, 50), centroid_fft_idx=positions,
                  n_rmu_logical=2, n_rmu_padded=4, coordinate_kind='fractional')
    receipt = WavefunctionBasisReceipt.from_source(**kwargs)
    bound = WavefunctionBasisReceipt.from_bound_source(**kwargs, wfn_fingerprint_binding=object())
    receipt.assert_same_carrier(bound, where='source binding')
    assert receipt.coordinate_kind == 'fractional'
    assert receipt.centroid_fingerprint_scheme == FRACTIONAL_CENTROID_TABLE_FINGERPRINT_SCHEME
    receipt.assert_matches_source(**kwargs, where='exact source')
    changed = positions.copy()
    changed[0, 0] += 1e-6
    with pytest.raises(ValueError, match='centroid_table_md5'):
        receipt.assert_matches_source(**dict(kwargs, centroid_fft_idx=changed), where='displaced source')
    with pytest.raises(ValueError, match='fingerprint scheme'):
        replace(receipt, coordinate_kind='fft_indices')
    legacy = WavefunctionBasisReceipt.from_source(**dict(kwargs, coordinate_kind='fft_indices',
                                                        centroid_fft_idx=np.array([[1, 2, 3], [4, 5, 6]])))
    assert legacy.centroid_fingerprint_scheme == CENTROID_TABLE_FINGERPRINT_SCHEME
    with pytest.raises(ValueError, match='physical source fields'):
        receipt.assert_same_source(legacy, where='coordinate type')


def _header(positions, **extra):
    return IsdfHeader.build(r_mu_crystal=positions, coordinate_kind='fractional',
                            fft_grid=(50, 50, 50), density='scalar', vertex_mu_L=0,
                            **extra)


def test_fractional_header_roundtrip_and_probe_have_no_fake_fft_indices(tmp_path):
    positions = np.array([[-.13, .27, 1.31], [.630001, .77, .81]])
    header = _header(positions)
    assert header.r_mu_fft_idx is None and header.n_rmu == 2
    path = tmp_path / 'zeta.h5'
    write_isdf_header(path, header, mode='w')
    with h5py.File(path, 'a') as f:
        assert 'r_mu_fft_idx' not in f['isdf_header/centroids']
        assert f['isdf_header/centroids/r_mu_crystal'].dtype == np.float64
        f.create_dataset('zeta_q_G', shape=(1, 2, 3), dtype=np.complex128)
    loaded = read_isdf_header(path)
    assert loaded.coordinate_kind == 'fractional' and loaded.r_mu_fft_idx is None
    np.testing.assert_array_equal(loaded.centroid_coordinates, positions)
    bound = SimpleNamespace()
    bind_isdf_attrs(bound, loaded)
    assert bound.coordinate_kind == 'fractional' and bound.n_rmu == 2
    np.testing.assert_array_equal(bound.centroid_coordinates, positions)
    probe = probe_zeta_file(path)
    assert probe.readable and probe.coordinate_kind == 'fractional'
    assert probe.r_mu_fft_idx is None and probe.mu_extent == 2
    np.testing.assert_array_equal(probe.centroid_coordinates, positions)


def test_legacy_header_type_is_default_and_invalid_type_never_infers_dtype(tmp_path):
    indices = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    path = tmp_path / 'legacy.h5'
    header = IsdfHeader.build(r_mu_fft_idx=indices, fft_grid=(8, 8, 8), density='scalar', vertex_mu_L=0)
    write_isdf_header(path, header, mode='w')
    with h5py.File(path, 'a') as f:
        del f['isdf_header/centroids/coordinate_kind']
    loaded = read_isdf_header(path)
    assert loaded.coordinate_kind == 'fft_indices'
    np.testing.assert_array_equal(loaded.centroid_coordinates, indices)
    assert probe_zeta_file(path).coordinate_kind == 'fft_indices'
    with h5py.File(path, 'a') as f:
        f['isdf_header/centroids'].create_dataset('coordinate_kind', data=np.bytes_('float64'))
    with pytest.raises(ValueError, match='coordinate_kind'):
        read_isdf_header(path)
    assert not probe_zeta_file(path).readable


def test_fractional_header_rejects_missing_positions_and_fabricated_indices(tmp_path):
    with pytest.raises(ValueError, match='r_mu_crystal'):
        _header(None)
    with pytest.raises(ValueError, match='must not carry'):
        _header([[.1, .2, .3]], r_mu_fft_idx=np.array([[5, 10, 15]]))
    path = tmp_path / 'poisoned.h5'
    write_isdf_header(path, _header([[.1, .2, .3]]), mode='w')
    with h5py.File(path, 'a') as f:
        f['isdf_header/centroids'].create_dataset('r_mu_fft_idx', data=[[5, 10, 15]])
    with pytest.raises(ValueError, match='must not carry'):
        read_isdf_header(path)
    assert not probe_zeta_file(path).readable


def test_fractional_header_rejects_lost_precision_and_missing_position_dataset(tmp_path):
    path = tmp_path / 'float32.h5'
    write_isdf_header(path, _header([[.123456789, .2, .3]]), mode='w')
    with h5py.File(path, 'a') as f:
        group = f['isdf_header/centroids']
        del group['r_mu_crystal']
        group.create_dataset('r_mu_crystal', data=np.array([[.123456789, .2, .3]], dtype=np.float32))
    with pytest.raises(ValueError, match='float64'):
        read_isdf_header(path)
    assert not probe_zeta_file(path).readable
    with h5py.File(path, 'a') as f:
        del f['isdf_header/centroids/r_mu_crystal']
    with pytest.raises(KeyError):
        read_isdf_header(path)
    assert not probe_zeta_file(path).readable

"""Logical restart bands are metadata, not a replicated wavefunction face."""
import h5py
import numpy as np
import pytest

from file_io.restart_bundle import read_metadata
from file_io.tagged_arrays import (
    BAND_WINDOW_SCHEMA_VERSION, RESTART_LOGICAL_SHAPE_ATTR,
    RESTART_CARRIER_SHAPE_ATTR, RESTART_PADDED_AXES_ATTR,
)


def _bundle(path, *, logical=5, carrier=8, disk=None, parents=1,
            mu=7, offset=10, paired=True, receipts=True, energy_disk=None,
            transverse=False):
    disk = logical if disk is None else disk
    energy_disk = disk if energy_disk is None else energy_disk
    window = np.array([offset+min(i, logical) for i in range(4)]
                      + [offset+logical], dtype=np.int64)
    with h5py.File(path, 'w') as f:
        f['lorrax_io_committed'] = np.array([1], np.uint8)
        f['band_window_schema'] = BAND_WINDOW_SCHEMA_VERSION
        f['band_window'] = window
        f['band_window_split'] = np.array([window[-1], window[-1]])
        if paired:
            f['band_window_carrier'] = np.r_[window[:4], offset+carrier]
        f['n_rmu_logical'] = mu
        f['kgrid'] = [max(1, parents), 1, 1]
        f['psi_parent_k_rows'] = np.arange(parents)
        f.create_dataset('enk_full', shape=(max(1, parents), energy_disk), dtype='f8')
        for suffix, width in (('', mu), ('_transverse', 3)):
            if suffix and not transverse:
                continue
            f.create_dataset('psi_parent_y'+suffix,
                             shape=(parents, disk, 2, width), dtype='c16')
            f.create_dataset('psi_parent_y'+suffix+'_mun' if suffix else 'psi_parent_y_mun',
                             shape=(parents, 2, width, disk), dtype='c16')
        if receipts:
            for name in ('psi_parent_y', 'psi_parent_y_mun', 'enk_full',
                         'psi_parent_y_transverse', 'psi_parent_y_transverse_mun'):
                if name not in f:
                    continue
                ds = f[name]
                axis = 3 if name.endswith('_mun') else 1
                physical = list(ds.shape); physical[axis] = carrier
                ds.attrs[RESTART_LOGICAL_SHAPE_ATTR] = ds.shape
                ds.attrs[RESTART_CARRIER_SHAPE_ATTR] = physical
                ds.attrs[RESTART_PADDED_AXES_ATTR] = [[axis, logical, carrier, 4]]
    return path


def _watch_psi(monkeypatch, *, forbid=False):
    original = h5py.Dataset.__getitem__
    reads = []
    def read(ds, key):
        if ds.name.rsplit('/', 1)[-1].startswith('psi_parent_y'):
            if forbid:
                raise AssertionError('metadata must not read modern wavefunction values')
            assert ds.name == '/psi_parent_y'
            assert isinstance(key, tuple) and len(key) == 4
            assert key[0] == 0 and isinstance(key[1], int) and isinstance(key[2], int)
            assert isinstance(key[3], slice)
            value = original(ds, key)
            assert value.nbytes <= 1 << 20
            reads.append((key, value.nbytes))
            return value
        return original(ds, key)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', read)
    return reads


def test_modern_metadata_never_reads_multigigabyte_parent_or_dense_family(monkeypatch, tmp_path):
    # Sparse HDF5 creation allocates no six-GB wavefunction payload. Reading
    # any of it fails the test, so the old psi_parent_y[0] scan cannot pass.
    large = _bundle(tmp_path/'large.h5', logical=20000, carrier=20000,
                    mu=10000, offset=0, receipts=False, transverse=True)
    reads = _watch_psi(monkeypatch, forbid=True)
    result = read_metadata(large)
    assert result['logical_band_count'] == 20000 and not reads
    assert result['energies'].shape == (1, 20000)
    assert result['family_shapes']['charge'] == (1, 20000, 2, 10000)
    assert result['family_shapes']['current'] == (1, 20000, 2, 3)


@pytest.mark.parametrize('disk,receipts', [(5, True), (8, False)])
def test_declared_logical_prefix_wins_over_zero_amplitudes_and_producer_padding(monkeypatch, tmp_path, disk, receipts):
    path = _bundle(tmp_path/'padded.h5', disk=disk, receipts=receipts, transverse=True)
    _watch_psi(monkeypatch, forbid=True)
    result = read_metadata(path)
    # b0=10: b4=15 is an absolute edge, not a count. Every psi is all zero.
    assert result['logical_band_count'] == 5
    assert result['band_window'].tolist() == [10, 11, 12, 13, 15]
    assert result['energies'].shape[1] == disk
    assert result['family_shapes']['charge'][1] == disk


def test_authenticated_empty_window_needs_no_first_parent_row(monkeypatch, tmp_path):
    path = _bundle(tmp_path/'empty.h5', logical=0, carrier=0,
                   parents=0, receipts=False)
    _watch_psi(monkeypatch, forbid=True)
    result = read_metadata(path)
    assert result['logical_band_count'] == 0
    assert result['energies'].shape == (1, 0)
    assert result['parent_k_rows'].size == 0


def test_appended_faces_keep_their_own_producer_padding(monkeypatch, tmp_path):
    path = _bundle(tmp_path/'portable.h5')
    with h5py.File(path, 'a') as f:
        # Original producer rounded five bands to eight. A later append on
        # a smaller mesh rounds the same logical disk faces to six instead;
        # energies and the original window still carry their old receipts.
        for name, axis in (('psi_parent_y', 1), ('psi_parent_y_mun', 3)):
            ds = f[name]
            physical = list(ds.shape); physical[axis] = 6
            ds.attrs[RESTART_CARRIER_SHAPE_ATTR] = physical
            ds.attrs[RESTART_PADDED_AXES_ATTR] = [[axis, 5, 6, 2]]
    _watch_psi(monkeypatch, forbid=True)
    assert read_metadata(path)['logical_band_count'] == 5


def test_legacy_last_populated_band_and_all_zero_fallback_are_bounded(monkeypatch, tmp_path):
    path = _bundle(tmp_path/'legacy.h5', paired=False, receipts=False,
                   logical=8, carrier=8, disk=8, parents=2, mu=131073)
    with h5py.File(path, 'a') as f:
        # Only the last centroid of one spinor in parent zero is populated.
        # Another parent's higher band must not change the historic answer.
        f['psi_parent_y'][0, 4, 1, -1] = 2j
        f['psi_parent_y'][1, 7, 0, 0] = 1
    reads = _watch_psi(monkeypatch)
    result = read_metadata(path)
    assert result['logical_band_count'] == 5
    assert reads and max(size for _, size in reads) <= 1 << 20
    # A nonfinite physical entry had counted as populated on the old route.
    with h5py.File(path, 'a') as f:
        f['psi_parent_y'][0, 7, 0, 0] = np.nan
    assert read_metadata(path)['logical_band_count'] == 8
    zero = _bundle(tmp_path/'legacy_zero.h5', paired=False, receipts=False,
                   logical=3, carrier=4, disk=4, energy_disk=6)
    assert read_metadata(zero)['logical_band_count'] == 6
    no_parent = _bundle(tmp_path/'legacy_no_parent.h5', paired=False, receipts=False,
                        logical=0, carrier=0, parents=0)
    with pytest.raises(ValueError, match='Legacy restart band metadata requires a parent'):
        read_metadata(no_parent)


@pytest.mark.parametrize('fault', ['fractional', 'reversed', 'carrier', 'psi_extent',
                                 'energy_extent', 'shape_receipt', 'partial_receipt',
                                 'band_axis', 'transverse'])
def test_torn_logical_and_storage_receipts_refuse_without_wavefunction_reads(monkeypatch, tmp_path, fault):
    path = _bundle(tmp_path/'bad.h5', transverse=True)
    with h5py.File(path, 'a') as f:
        if fault == 'fractional':
            del f['band_window']
            f['band_window'] = np.array([10., 11., 12., 13., 15.5])
        elif fault == 'reversed':
            f['band_window'][1] = 9
        elif fault == 'carrier':
            f['band_window_carrier'][1] = 12
        elif fault in ('psi_extent', 'energy_extent', 'transverse'):
            name = {'psi_extent': 'psi_parent_y', 'energy_extent': 'enk_full',
                    'transverse': 'psi_parent_y_transverse'}[fault]
            shape = list(f[name].shape); shape[1] = 6
            del f[name]
            f.create_dataset(name, shape=shape, dtype='f8' if name == 'enk_full' else 'c16')
        elif fault == 'shape_receipt':
            f['psi_parent_y'].attrs[RESTART_LOGICAL_SHAPE_ATTR] = [1, 4, 2, 7]
        elif fault == 'partial_receipt':
            del f['psi_parent_y'].attrs[RESTART_LOGICAL_SHAPE_ATTR]
        else:
            # Self-consistent dataset receipt that contradicts the physical
            # window: declaring eight real bands cannot turn its pad into ψ.
            del f['psi_parent_y']
            ds = f.create_dataset('psi_parent_y', shape=(1, 8, 2, 7), dtype='c16')
            ds.attrs[RESTART_LOGICAL_SHAPE_ATTR] = ds.shape
            ds.attrs[RESTART_CARRIER_SHAPE_ATTR] = ds.shape
            ds.attrs[RESTART_PADDED_AXES_ATTR] = [[1, 8, 8, 4]]
    _watch_psi(monkeypatch, forbid=True)
    with pytest.raises(ValueError, match='band-window|band receipts|band extent|storage shape|band-axis receipt'):
        read_metadata(path)

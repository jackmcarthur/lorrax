"""Admission and family contracts of the single post-landing bundle reader."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from file_io import restart_bundle as bundle


def _current(path, ns):
    with h5py.File(path, 'w') as f:
        f['psi_parent_y'] = np.zeros((2, 3, ns, 5), complex)
        f['psi_parent_y_mun'] = np.zeros((2, ns, 5, 3), complex)
        f['psi_parent_k_rows'] = [0, 2]
        f['band_window'] = [0, 0, 1, 3, 3]
        f['band_window_schema'] = 2
        f['enk_full'] = np.zeros((4, 3))
        f['kgrid'] = [2, 2, 1]
        f['n_rmu_logical'] = 5
        f['V_qmunu'] = np.zeros((4, 5, 5), complex)
        f['W0_qmunu'] = np.zeros((4, 5, 5), complex)
        f['W0_qmunu'].attrs['W0_ready'] = False
    return path


@pytest.mark.parametrize('ns', [1, 2, 4])
def test_family_reports_true_spin_extent(tmp_path, ns):
    path = _current(tmp_path / 'current.h5', ns)
    assert bundle.read_metadata(path)['family_shapes']['charge'] == (2, 3, ns, 5)


@pytest.mark.parametrize('reader', [bundle.read_metadata,
                                  bundle.read_downfold_geometry,
                                  bundle.require_screened_bundle])
def test_old_full_wavefunction_bundle_refused_once(tmp_path, reader):
    path = _current(tmp_path / 'old.h5', 2)
    with h5py.File(path, 'a') as f:
        f['psi_full_y'] = np.zeros((4, 3, 2, 5), complex)
    with pytest.raises(ValueError) as exc:
        reader(path)
    assert str(exc.value) == (
        'this bundle predates the raw-parent format; '
        'regenerate it with gwjax at main >= 891047f4')


def test_missing_parent_face_is_not_derived(tmp_path):
    path = _current(tmp_path / 'partial.h5', 4)
    with h5py.File(path, 'a') as f:
        del f['psi_parent_y_mun']
    with pytest.raises(ValueError, match='regenerate it with gwjax'):
        bundle.read_metadata(path)


def test_unfinished_screening_refused_before_transport(tmp_path):
    path = _current(tmp_path / 'unfinished.h5', 4)
    with pytest.raises(ValueError, match='not persisted'):
        bundle.read_interaction(path, 'screened', None)
    with pytest.raises(ValueError, match='PLACEHOLDER'):
        bundle.read_downfold_geometry(path)


def test_nohead_tensor_uses_screening_commit_receipt(tmp_path, monkeypatch):
    path = _current(tmp_path / 'ready.h5', 2)
    with h5py.File(path, 'a') as f:
        f['W0_qmunu'].attrs['W0_ready'] = True
        f['W0_qmunu_nohead'] = np.zeros((4, 5, 5), complex)
    sentinel = object()
    def transport(filename, dataset, mesh):
        assert Path(filename) == path
        assert dataset == 'W0_qmunu_nohead'
        return sentinel
    monkeypatch.setattr(bundle, 'read_munu_tensor_from_h5', transport)
    assert bundle.read_interaction(path, 'screened', None, nohead=True) is sentinel

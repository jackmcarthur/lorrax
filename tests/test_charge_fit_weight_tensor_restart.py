"""Complete tensor restarts authenticate the loss before their large reads."""
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from gw import gw_init
from file_io.tagged_arrays import (
    CHARGE_ZETA_IDENTITY_DATASET, CHARGE_ZETA_PROVENANCE_DATASET,
)


class _ReachedTensorLoader(Exception):
    pass


@pytest.fixture
def bound_wfn(monkeypatch):
    import common.parallel_transport as transport
    # Isolate the fitting policy; actual-WFN source joins run separately.
    monkeypatch.setattr(transport, 'wfn_fingerprint', lambda _: 'same-source-wfn')
    occupied = np.zeros((1, 2, 7))
    occupied[..., :2] = 1.0
    return SimpleNamespace(occs=occupied)


def _provenance(weight=1.0, stop=2):
    record = {'schema': 2, 'band_range_left_logical': [0, 4],
              'band_range_right_logical': [0, 7], 'logical_band_stop': 7}
    if weight != 1.0:
        record['charge_fit_endpoint_weights'] = {
            'schema': 'occupied_band_endpoints_v1', 'occupied_stop': stop,
            'occupied_weight': float(weight), 'empty_weight': 1.0}
    return json.dumps(record, sort_keys=True)


def _records(path, wfn, provenance, *, embedded=True, identified=True):
    with h5py.File(path, 'w') as stream:
        if identified:
            identity = gw_init.charge_zeta_identity(provenance, wfn=wfn)
            stream.create_dataset(CHARGE_ZETA_IDENTITY_DATASET,
                                  data=np.asarray(tuple(identity.values()), dtype='S'))
        if embedded:
            stream.create_dataset(CHARGE_ZETA_PROVENANCE_DATASET,
                                  data=np.bytes_(provenance.encode('utf-8')))
        # No storage is allocated for these large logical arrays.
        stream.create_dataset('V_qmunu', shape=(16, 100000, 100000),
                              chunks=(1, 10, 10), dtype='complex128')


def _restart(path, tmp_path, wfn, *, weight=1.0, stop=2, sigma_stop=3):
    cfg = SimpleNamespace(
        paths=SimpleNamespace(atomic_reconstruction_dir=None),
        backend=SimpleNamespace(zeta_occupied_weight=weight),
        occ_smearing_width_ry=None,
        screening=SimpleNamespace(occ_broadening_ev=0.0))
    bands = SimpleNamespace(b0=0, b1=0, b2=stop, b3=sigma_stop, b4=7)
    def refuse_large_read(*args, **kwargs):
        raise _ReachedTensorLoader
    return gw_init._read_authenticated_restart(
        lambda array, axes: array, bands, cfg, refuse_large_read,
        None, SimpleNamespace(n_rmu=2), lambda *_: None, str(path),
        charge_fit_context={'wfn': wfn, 'wfn_fingerprint_binding': None,
                            'tmp_dir': str(tmp_path)})


@pytest.mark.parametrize('weight', [1.0, 4.0])
@pytest.mark.parametrize('sigma_stop', [3, 6])
def test_same_loss_preserves_original_training_and_sigma_only_window_changes(
        tmp_path, bound_wfn, weight, sigma_stop):
    path = tmp_path / 'restart.h5'
    _records(path, bound_wfn, _provenance(weight))
    with pytest.raises(_ReachedTensorLoader):
        _restart(path, tmp_path, bound_wfn, weight=weight, sigma_stop=sigma_stop)


@pytest.mark.parametrize('old,new', [(1.0, 4.0), (4.0, 1.0), (4.0, 2.0)])
def test_changed_loss_refuses_before_loading_tensor_arrays(tmp_path, bound_wfn, old, new):
    path = tmp_path / 'restart.h5'
    _records(path, bound_wfn, _provenance(old))
    with pytest.raises(ValueError, match='charge_fit_endpoint_weights changed'):
        _restart(path, tmp_path, bound_wfn, weight=new)


def test_changed_occupied_boundary_refuses_before_tensor_read(tmp_path, bound_wfn):
    path = tmp_path / 'restart.h5'
    _records(path, bound_wfn, _provenance(4.0))
    bound_wfn.occs[..., 2] = 1.0
    with pytest.raises(ValueError, match='charge_fit_endpoint_weights changed'):
        _restart(path, tmp_path, bound_wfn, weight=4.0, stop=3)


def test_poisoned_embedded_provenance_refuses_its_own_receipt(tmp_path, bound_wfn):
    path = tmp_path / 'restart.h5'
    _records(path, bound_wfn, _provenance(4.0))
    with h5py.File(path, 'a') as stream:
        del stream[CHARGE_ZETA_PROVENANCE_DATASET]
        stream.create_dataset(CHARGE_ZETA_PROVENANCE_DATASET,
                             data=np.bytes_(_provenance(2.0).encode('utf-8')))
    with pytest.raises(ValueError, match='authoritative WFN-bound'):
        _restart(path, tmp_path, bound_wfn, weight=4.0)


@pytest.mark.parametrize('case', ['matched', 'changed', 'unfinished', 'missing'])
def test_identified_legacy_fallback_requires_the_completed_matching_zeta_header(
        tmp_path, bound_wfn, monkeypatch, case):
    import file_io.restart_bundle as restart
    path = tmp_path / 'restart.h5'
    _records(path, bound_wfn, _provenance(4.0), embedded=False)
    if case != 'missing':
        (tmp_path / 'zeta_q.h5').touch()
    monkeypatch.setattr(restart, 'read_isdf_header', lambda _: SimpleNamespace(
        zeta_is_done=case != 'unfinished',
        fit_provenance=_provenance(2.0 if case == 'changed' else 4.0)))
    expected = _ReachedTensorLoader if case == 'matched' else ValueError
    with pytest.raises(expected):
        _restart(path, tmp_path, bound_wfn, weight=4.0)


def test_unidentified_legacy_bundle_only_serves_the_default_loss(tmp_path, bound_wfn):
    path = tmp_path / 'restart.h5'
    _records(path, bound_wfn, _provenance(), embedded=False, identified=False)
    with pytest.raises(_ReachedTensorLoader):
        _restart(path, tmp_path, bound_wfn)
    with pytest.raises(ValueError, match='legacy restart has no authenticated'):
        _restart(path, tmp_path, bound_wfn, weight=4.0)


def test_source_identity_change_refuses_even_when_loss_agrees(
        tmp_path, bound_wfn, monkeypatch):
    import common.parallel_transport as transport
    path = tmp_path / 'restart.h5'
    _records(path, bound_wfn, _provenance(4.0))
    monkeypatch.setattr(transport, 'wfn_fingerprint', lambda _: 'different-source-wfn')
    with pytest.raises(ValueError, match='authoritative WFN-bound'):
        _restart(path, tmp_path, bound_wfn, weight=4.0)

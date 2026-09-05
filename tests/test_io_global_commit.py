"""Failure receipts carry the remote error; incomplete files refuse on restart."""
import json
from unittest.mock import Mock

import h5py
import numpy as np
import pytest

from common import collectives
from file_io.commit_state import COMMIT_STATE, assert_committed, set_commit_state


def test_remote_data_error_refuses_with_same_receipt(monkeypatch):
    monkeypatch.setattr(collectives, 'process_rank', lambda: 2)
    remote = collectives._io_error_receipt(OSError('write failed'))
    good = collectives._io_error_receipt(None)
    monkeypatch.setattr(collectives, 'all_gather_processes',
                        lambda x: np.stack([good, good, remote, good]))
    with pytest.raises(RuntimeError, match='failing rank=2; OSError: write failed'):
        collectives.agree_io_error(None, path='/private/restart.h5', stage='data_close')


def test_receipts_are_bounded_even_for_unicode_errors():
    receipt = collectives._io_error_receipt(OSError('🌋' * 10000))
    assert receipt.shape == (4096,)
    assert len(json.loads(bytes(receipt).rstrip(b'\0'))[-1]) == 240


def test_rank0_transaction_stops_after_preflight_failure():
    write = Mock()
    def validate():
        raise ValueError('bad replicated shape')
    with pytest.raises(RuntimeError, match='preflight.*bad replicated shape'):
        collectives.rank0_transaction('/tmp/output.h5', stage='test',
                                      validate=validate, write=write)
    write.assert_not_called()


def test_rank0_filesystem_error_names_action_and_path():
    def write():
        raise OSError('disk full')
    with pytest.raises(RuntimeError, match='path=/private/qp.h5; stage=QP; failing rank=0'):
        collectives.rank0_transaction('/private/qp.h5', stage='QP', write=write)


def test_incomplete_restart_refuses_even_with_stale_ready_flag(tmp_path):
    from bse.bse_loading import _refuse_unpersisted
    path = tmp_path / 'restart.h5'
    with h5py.File(path, 'w') as f:
        ds = f.create_dataset('V_qmunu', data=np.zeros((1, 2, 2)))
        ds.attrs['V_ready'] = True
        set_commit_state(f, False)
    with h5py.File(path) as f:
        with pytest.raises(ValueError, match='not globally committed'):
            _refuse_unpersisted(f['V_qmunu'], 'V_qmunu', str(path))
    with h5py.File(path, 'a') as f:
        set_commit_state(f, True)
        assert_committed(f)
        assert int(f[COMMIT_STATE][0]) == 1


def test_payload_comparator_detects_changed_bytes_and_attributes(tmp_path):
    from bench.compare_io_artifacts import compare
    left, right = tmp_path / 'left.h5', tmp_path / 'right.h5'
    for path in (left, right):
        with h5py.File(path, 'w') as f:
            f.create_dataset('data', data=np.array([0.0, 1.0]))
    assert not compare(left, right)['differences']
    with h5py.File(right, 'a') as f:
        f['data'][0] = -0.0  # numerically equal, bitwise different
    assert compare(left, right)['differences'][0]['payload'] == 'data'
    with h5py.File(right, 'a') as f:
        f['data'][0] = 0.0
        f['data'].attrs['provenance'] = 'changed'
    assert compare(left, right)['differences'][0]['attributes'] == 'data'

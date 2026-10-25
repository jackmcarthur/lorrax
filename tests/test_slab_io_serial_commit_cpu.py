"""The CPU emulation tier obeys SlabIO's persistent transaction contract."""
import h5py
import numpy as np
import pytest

import runtime  # noqa: F401
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from file_io.commit_state import read_commit_state, set_commit_state
from file_io.slab_io import SlabIO


def _mesh():
    devices = jax.devices('cpu')
    if len(devices) < 4 or jax.process_count() != 1:
        pytest.skip('requires four emulated CPU devices in one process')
    return Mesh(np.asarray(devices[:4]).reshape(2, 2), ('x', 'y'))


def test_serial_commit_follows_data_and_deferred_group_metadata(tmp_path):
    path, mesh = str(tmp_path / 'transaction.h5'), _mesh()
    io = SlabIO(path, mesh=mesh, mode='w')
    assert io._backend._h5['lorrax_io_committed'][0] == 0
    io.create_dataset('W0_lr/Ft', shape=(2, 3), dtype=np.complex128)
    io.write_attr('W0_lr/P', np.eye(2))
    io.stamp_dataset_attrs('W0_lr', {'role': 'channels'})
    io.close()
    assert read_commit_state(path) == 1
    with h5py.File(path, 'r') as h5:
        assert h5['W0_lr'].attrs['role'] == 'channels'
        np.testing.assert_array_equal(h5['W0_lr/P'][()], np.eye(2))
    # An append invalidates even without any queued metadata, and must still
    # publish success. This was missing in the serial tier.
    with SlabIO(path, mesh=mesh, mode='a') as io:
        assert io._backend._h5['lorrax_io_committed'][0] == 0
    assert read_commit_state(path) == 1
    before = open(path, 'rb').read()
    with SlabIO(path, mesh=mesh, mode='r') as io:
        assert io.read_small('W0_lr/P').shape == (2, 2)
    assert open(path, 'rb').read() == before


def test_serial_authenticates_receipts_but_accepts_legacy_files(tmp_path):
    path, mesh = str(tmp_path / 'legacy.h5'), _mesh()
    with h5py.File(path, 'w') as h5:
        h5['old'] = [3]
    with SlabIO(path, mesh=mesh, mode='r'):
        pass
    assert read_commit_state(path) is None
    with SlabIO(path, mesh=mesh, mode='a'):
        pass
    assert read_commit_state(path) == 1
    with h5py.File(path, 'a') as h5:
        set_commit_state(h5, False)
    for mode in ('r', 'a'):
        with pytest.raises(ValueError, match='GATE io_global_commit'):
            SlabIO(path, mesh=mesh, mode=mode)
    # Failed opens release their owner claim and file handle.
    with h5py.File(path, 'a') as h5:
        set_commit_state(h5, True)
    with pytest.raises(RuntimeError, match="mode='r' acquired deferred"):
        with SlabIO(path, mesh=mesh, mode='r') as io:
            io.write_attr('forbidden', np.int32(7))
    with h5py.File(path, 'r') as h5:
        assert 'forbidden' not in h5
    assert read_commit_state(path) == 1


def test_serial_body_or_metadata_failure_never_commits_on_second_close(tmp_path):
    mesh = _mesh()
    path = str(tmp_path / 'body.h5')
    with pytest.raises(RuntimeError, match='body failed'):
        with SlabIO(path, mesh=mesh, mode='w') as io:
            io.write_attr('not_published', np.int32(1))
            raise RuntimeError('body failed')
    io.close()
    assert read_commit_state(path) == 0
    with h5py.File(path, 'r') as h5:
        assert 'not_published' not in h5
    path = str(tmp_path / 'metadata.h5')
    io = SlabIO(path, mesh=mesh, mode='w')
    io.write_attr('partial', np.int32(2))
    io.stamp_dataset_attrs('missing', {'must_fail': True})
    with pytest.raises(RuntimeError, match='GATE io_global_commit:.*missing'):
        io.close()
    io.close()
    assert read_commit_state(path) == 0
    with pytest.raises(ValueError, match='GATE io_global_commit'):
        SlabIO(path, mesh=mesh, mode='r')


def test_metadata_materialization_failure_precedes_publication(tmp_path, monkeypatch):
    from file_io import _slab_io_ffi as transport
    path, mesh = str(tmp_path/'materialize.h5'), _mesh()
    io = SlabIO(path, mesh=mesh, mode='w')
    io.write_attr('must_not_land', np.int32(3))
    io.stamp_dataset_attrs('missing', {'unmaterializable': object()})

    def failed_materialization(_value):
        raise TypeError('injected metadata materialization failure')

    with monkeypatch.context() as guard:
        guard.setattr(transport, '_host_attr_value', failed_materialization)
        with pytest.raises(RuntimeError, match='metadata materialization failure'):
            io.close()
    io.close()
    assert io._backend._owner_token is None
    assert read_commit_state(path) == 0
    with h5py.File(path, 'r') as h5:
        assert 'must_not_land' not in h5
    # Failed publication does not leave a live owner or poison another file.
    with SlabIO(str(tmp_path/'independent.h5'), mesh=mesh, mode='w'):
        pass
    assert read_commit_state(tmp_path/'independent.h5') == 1


def test_serial_data_close_failure_leaves_receipt_incomplete(tmp_path):
    path, mesh = str(tmp_path / 'close.h5'), _mesh()
    io = SlabIO(path, mesh=mesh, mode='w')
    actual = io._backend._h5

    class FailedClose:
        def close(self):
            actual.close()
            raise OSError('injected H5Fclose failure')

    io._backend._h5 = FailedClose()
    with pytest.raises(OSError, match='H5Fclose'):
        io.close()
    io.close()
    assert io._backend._owner_token is None
    assert read_commit_state(path) == 0


def test_serial_partial_write_error_stays_failed_when_caught_in_body(tmp_path):
    path, mesh = str(tmp_path / 'partial.h5'), _mesh()
    io = SlabIO(path, mesh=mesh, mode='w')
    data = jax.device_put(np.ones((4, 4)), NamedSharding(mesh, P('x', 'y')))
    io.create_dataset('A', shape=data.shape, dtype=data.dtype)
    actual, writes = io._backend._write_block, []

    def fail_second(*args):
        writes.append(1)
        if len(writes) == 2:
            raise OSError('injected second shard failure')
        return actual(*args)

    io._backend._write_block = fail_second
    with pytest.raises(OSError, match='second shard'):
        with io:
            with pytest.raises(OSError, match='second shard'):
                io.write_slab('A', data)
    assert len(writes) == 2
    assert read_commit_state(path) == 0
    with h5py.File(path, 'r') as h5:
        assert 0 < np.count_nonzero(h5['A'][()]) < data.size

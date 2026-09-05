"""Production rank-0 writers must deliver filesystem errors to their callers."""
import h5py
import numpy as np
import pytest

from file_io import tagged_arrays
from file_io.commit_state import assert_committed, set_commit_state


def test_head_writer_h5py_failure_names_file_and_stage(tmp_path, monkeypatch):
    path = tmp_path / 'restart.h5'
    def broken_open(*args, **kwargs):
        raise OSError('injected h5py failure')
    monkeypatch.setattr(tagged_arrays.h5py, 'File', broken_open)
    with pytest.raises(RuntimeError) as raised:
        tagged_arrays.write_head_scalars_to_h5(str(path), vhead=1.0)
    message = str(raised.value)
    assert str(path) in message
    assert 'stage=restart.head_scalars; failing rank=0' in message
    assert 'injected h5py failure' in message


def test_head_writer_invalid_shape_does_not_mutate_committed_artifact(tmp_path):
    path = tmp_path / 'restart.h5'
    with h5py.File(path, 'w') as f:
        set_commit_state(f, True)
        f.create_dataset('vhead', data=2.0)
    with pytest.raises(ValueError):
        tagged_arrays.write_head_scalars_to_h5(str(path), vhead=3.0, S_cart=np.ones(8))
    with h5py.File(path) as f:
        assert_committed(f)
        assert float(f['vhead'][()]) == 2.0

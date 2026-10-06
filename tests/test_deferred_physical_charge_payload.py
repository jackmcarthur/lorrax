"""Physical charge publication drains its payload before certifying the file."""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest


@pytest.fixture
def pending_charge(tmp_path, monkeypatch):
    from common import collectives
    from file_io import slab_io
    from gw import v_q_g_flat

    path = tmp_path / 'charge.h5'
    with h5py.File(path, 'w') as stream:
        stream.create_group('isdf_header').create_dataset('zeta_is_done', data=False)
    state = dict(closed=False)
    smooth = np.arange(12).reshape(1, 3, 4).astype(np.complex128)
    delta = np.full(smooth.shape, .125+.25j)

    class QueuedWriter:
        def __init__(self, target, *, mode, mesh):
            assert target == str(path) and mode == 'a'
            self.payload = None

        def __enter__(self):
            return self

        def create_dataset(self, name, *, shape, dtype):
            assert name == 'zeta_q_G' and shape == smooth.shape
            assert dtype == np.complex128

        def __exit__(self, kind, value, traceback):
            if kind is None:
                with h5py.File(path, 'a') as stream:
                    stream['zeta_q_G'] = self.payload
            state['closed'] = True

    def root_transaction(target, *, stage, write):
        assert state['closed'], 'completion ran before collective writer close'
        assert stage == 'augmented_zeta_payload_complete'
        write()

    monkeypatch.setattr(slab_io, 'SlabIO', QueuedWriter)
    monkeypatch.setattr(collectives, 'rank0_transaction', root_transaction)

    def contract(v, *, keep, zeta_io):
        with h5py.File(path) as stream:
            assert not bool(stream['isdf_header/zeta_is_done'][()])
            assert 'isdf_header/fit_provenance' not in stream
        zeta_io.payload = smooth+delta
        return np.eye(3)

    zeta = SimpleNamespace(path=str(path), store=SimpleNamespace(Q=1),
        n_rmu=3, ngkmax=4, local_augmentation={}, pending_zeta_write=True,
        pending_fit_provenance='{"physical_charge":true}', contract_v=contract)
    return v_q_g_flat, zeta, path, smooth+delta


def test_deferred_file_contains_physical_density_after_writer_close(pending_charge):
    owner, zeta, path, physical = pending_charge
    result = owner._contract_live_charge(zeta, np.ones((1, 4)), keep=np.zeros((1, 1)), mesh=None)
    np.testing.assert_array_equal(result, np.eye(3))
    with h5py.File(path) as stream:
        np.testing.assert_array_equal(stream['zeta_q_G'][()], physical)
        assert bool(stream['isdf_header/zeta_is_done'][()])
        assert bytes(stream['isdf_header/fit_provenance'][()]) == b'{"physical_charge":true}'
    assert not zeta.pending_zeta_write and zeta.pending_fit_provenance is None


def test_failed_physical_stream_keeps_file_incomplete(pending_charge):
    owner, zeta, path, _ = pending_charge

    def fail(*args, **kwargs):
        raise OSError('injected payload failure')

    zeta.contract_v = fail
    with pytest.raises(OSError, match='injected payload failure'):
        owner._contract_live_charge(zeta, np.ones((1, 4)), keep=np.zeros((1, 1)), mesh=None)
    with h5py.File(path) as stream:
        assert not bool(stream['isdf_header/zeta_is_done'][()])
        assert 'fit_provenance' not in stream['isdf_header']
    assert zeta.pending_zeta_write


@pytest.mark.parametrize('missing', ['local_augmentation', 'pending_fit_provenance'])
def test_deferred_file_refuses_missing_physical_owner(pending_charge, missing):
    owner, zeta, path, _ = pending_charge
    setattr(zeta, missing, None)
    with pytest.raises(ValueError, match='requires'):
        owner._contract_live_charge(zeta, np.ones((1, 4)), keep=np.zeros((1, 1)), mesh=None)
    with h5py.File(path) as stream:
        assert not bool(stream['isdf_header/zeta_is_done'][()])


def test_ordinary_live_charge_retains_its_existing_call():
    from gw.v_q_g_flat import _contract_live_charge
    expected = object()
    zeta = SimpleNamespace(contract_v=lambda v, *, keep: expected)
    assert _contract_live_charge(zeta, None, keep=None, mesh=None) is expected

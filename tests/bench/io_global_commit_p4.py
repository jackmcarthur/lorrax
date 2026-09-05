"""Real P4 queued-write and rank-0 h5py failures, with per-rank exit receipts.

Run under lx -N 1 -G 4 -n 4, one mode per supervised process group. An injected
queued task raises AFTER its real collective H5Dwrite returns, so every peer
can finish H5Fclose. This tests the async Python writer error channel, not an
unrecoverable rank death inside MPI. Successful runs exercise the same task.
"""
import argparse
import json
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['marker_failure', 'slab_failure', 'metadata_failure', 'rank0_failure', 'success', 'restart'])
parser.add_argument('directory')
args = parser.parse_args()

from runtime import initialize_communicator_stack
runtime = initialize_communicator_stack()
import h5py
import jax
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from common.collectives import rank0_transaction
from file_io.slab_io import SlabIO
from file_io.commit_state import COMMIT_STATE

rank = jax.process_index()
assert jax.process_count() == 4
out = Path(args.directory)
path = out / 'slab.h5'
# Poison the newly-created receipt to prove zero initialization is explicit.
if args.mode == 'slab_failure':
    from file_io._slab_io_ffi import _FfiBackend
    original_create = _FfiBackend.create_dataset
    def poison_receipt(self, name, **kwargs):
        original_create(self, name, **kwargs)
        if name == COMMIT_STATE:
            count = int(self.mesh.size)
            poison = jax.make_array_from_callback(
                (count,), NamedSharding(self.mesh, P(tuple(self.mesh.axis_names))),
                lambda index: np.ones((count,), dtype=np.int32)[index])
            self.write_slab(name, poison)
            self._drain_pending()
    _FfiBackend.create_dataset = poison_receipt
elif args.mode == 'marker_failure':
    from common.async_io import AsyncDispatcher
    original_submit = AsyncDispatcher.submit
    def fail_initialization(self, task):
        def queued():
            task()
            if rank == 1:
                raise OSError('injected receipt initialization write failure')
        original_submit(self, queued)
    AsyncDispatcher.submit = fail_initialization

started = time.monotonic()
error = None
try:
    if args.mode == 'restart':
        # A later process must refuse, independent of its predecessor's exit.
        from bse.bse_loading import _refuse_unpersisted
        with h5py.File(path, 'r') as f:
            assert 'charge_zeta_identity' not in f
            assert 'V_ready' not in f['V_qmunu'].attrs
            assert int(f[COMMIT_STATE][0]) == 0
            _refuse_unpersisted(f['V_qmunu'], 'V_qmunu', str(path))
        raise AssertionError('BSE accepted failed restart')
    elif args.mode == 'rank0_failure':
        def write():
            # Actual h5py failure: a directory cannot be opened as a file.
            with h5py.File(out, 'w'):
                pass
        rank0_transaction(str(out), stage='rank0.h5py_write', write=write)
        raise AssertionError('writer failure did not stop next physics collective')
    else:
        mesh = runtime.mesh
        sharding = NamedSharding(mesh, P('x', 'y'))
        data = jax.make_array_from_callback(
            (8, 8), sharding, lambda index: np.arange(64, dtype=np.float64).reshape(8, 8)[index])
        with SlabIO(str(path), mode='w', mesh=mesh) as io:
            io.create_dataset('V_qmunu', shape=(5, 7), dtype=np.float64,
                              attrs={'V_ready': True, 'checksum': jax.numpy.sum(data)})
            io.write_attr('charge_zeta_identity', np.arange(3, dtype=np.int64))
            original_submit = io._backend._dispatcher.submit
            def submit(task):
                def queued():
                    task()
                    if rank == 1 and args.mode == 'slab_failure':
                        raise OSError('injected queued slab write failure')
                original_submit(queued)
            io._backend._dispatcher.submit = submit
            io.write_slab('V_qmunu', data)
            if args.mode == 'metadata_failure':
                # Invalid HDF5 attribute data fails only in the rank-0 reopen.
                io._backend._deferred_ds_attrs.append(('V_qmunu', {'bad': object()}))
        if args.mode != 'success':
            raise AssertionError('failed slab was published')
        with h5py.File(path, 'r') as f:
            np.testing.assert_array_equal(f['V_qmunu'][:], np.arange(64).reshape(8, 8)[:5, :7])
            assert int(f[COMMIT_STATE][0]) == 1
            assert bool(f['V_qmunu'].attrs['V_ready'])
except (RuntimeError, ValueError) as exc:
    error = str(exc)
    assert 'GATE io_global_commit' in error, error
    assert str(path if args.mode != 'rank0_failure' else out) in error
    if args.mode in {'slab_failure', 'marker_failure'}:
        assert 'stage=SlabIO.data_close; failing rank=1' in error
        if args.mode == 'marker_failure':
            with h5py.File(path, 'r') as f:
                assert int(f[COMMIT_STATE][0]) == 0
                assert 'V_qmunu' not in f
    elif args.mode == 'metadata_failure':
        assert 'stage=SlabIO.metadata_commit; failing rank=0' in error
    elif args.mode == 'rank0_failure':
        assert 'stage=rank0.h5py_write; failing rank=0' in error
    elif args.mode == 'success':
        raise
elapsed = time.monotonic() - started
assert elapsed < 60, elapsed
receipt = {'rank': rank, 'mode': args.mode, 'error': error,
           'elapsed_seconds': elapsed, 'exit_code': 17 if error else 0}
(out / f'{args.mode}.rank{rank}.json').write_text(json.dumps(receipt, indent=2))
print('IOCOMMIT_RECEIPT ' + json.dumps(receipt), flush=True)
# Allow the supervisor to observe all four nonzero exits, without the
# production exception hook racing the receipt writes on other processes.
if error:
    jax.distributed.shutdown()
    raise SystemExit(17)

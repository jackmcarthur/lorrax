"""Native async-read ordering against loader-owned h5py metadata readers."""


def main():
    import argparse
    import os
    from pathlib import Path
    import runtime
    stack = runtime.initialize_communicator_stack()
    import h5py
    import jax
    import numpy as np
    from jax.sharding import PartitionSpec as P
    from common.collectives import barrier
    from file_io import _slab_io_ffi as sff
    from file_io.slab_io import SlabIO
    from symmetry_maps import density_symmetry_check as check
    from wfn_loader import WfnLoader

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    rank = jax.process_index()
    path = args.out/'async_source.h5'
    shape = (2*int(stack.mesh.size), 2, 1 << 20, 2)
    if rank == 0:
        with h5py.File(path, 'w') as file:
            file.create_dataset('data', shape=shape, dtype=np.float64, fillvalue=4.25)
    barrier('ordering-fixture')

    entries = []
    def check_host_entry(kind):
        assert not sff._LANE._reads, f'{kind} entered h5py before native read completion'
        assert sff._LANE._owner is None
        entries.append(kind)

    class DatasetProbe:
        def __init__(self, dataset, kind):
            self.dataset, self.kind = dataset, kind
            self.shape, self.dtype = dataset.shape, dataset.dtype
        def __getitem__(self, index):
            check_host_entry(self.kind)
            return self.dataset[index]

    class FileProbe:
        def __init__(self, file):
            self.file = file
        def __getitem__(self, name):
            check_host_entry('dataset-open')
            return DatasetProbe(self.file[name], name)
        @property
        def attrs(self):
            check_host_entry('attrs')
            return self.file.attrs
        def close(self):
            check_host_entry('close')
            return self.file.close()

    fixture = Path(__file__).parent/'hsuite/fixture_na/WFN.h5'
    wfn = WfnLoader(fixture, mesh=stack.mesh, backend='phdf5')
    wfn._file = FileProbe(wfn._file)
    wfn._gvecs_raw = DatasetProbe(wfn._gvecs_raw, 'raw-G')
    pending_seen = []
    with SlabIO(path, mode='r', mesh=stack.mesh) as io:
        def post():
            value = io.read_slabs('data', shape=shape,
                offsets=np.zeros((1, 4), dtype=np.int64),
                valid_shapes=np.asarray([shape], dtype=np.int64),
                partition_spec=P(('x', 'y'), None, None, None), window_axis=2)
            assert sff._LANE._reads, 'the native completion token was not registered'
            pending_seen.append(any(not token.is_ready()
                for group in sff._LANE._reads.values() for token in group))
            return value

        def complete(value):
            value.block_until_ready()
            for shard in value.addressable_shards:
                assert np.all(np.asarray(shard.data) == 4.25)
            value.delete()

        value = post()
        wfn.get_gvec_nk(0)
        complete(value)
        value = post()
        check._content_digest(wfn)
        complete(value)
        value = post()
        check._raw_ibz_psi_k(wfn, 0, 3)
        complete(value)
        value = post()
        wfn.load_process_local(bands=(0, 3), k='ibz').block_until_ready()
        complete(value)
        value = post()
        wfn._qe_symmetry_checked = False
        wfn.resolve_qe_symmetry()
        complete(value)
        value = post()
        wfn.close()
        complete(value)
    # CPU dispatch can complete before returning; CUDA must exercise a live token.
    if jax.default_backend() == 'gpu':
        assert any(pending_seen), 'no in-flight native read exercised the ordering guard'
    assert wfn._parent_g_host_cache is None
    print(f'WFN_HDF5_ORDERING PASS rank={rank} pending={pending_seen} host_entries={len(entries)}', flush=True)
    runtime.finalize_process(0)


def test_requires_distributed_launcher():
    import pytest
    pytest.skip('run directly under the configured MPI launcher')


if __name__ == '__main__':
    main()

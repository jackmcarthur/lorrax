"""Independent raw-WFN FFT-index oracle and bounded-request cache contract."""


def main():
    import os
    import runtime
    stack = runtime.initialize_communicator_stack()
    import h5py
    import jax
    import numpy as np
    from wfn_loader import IBZRows, WfnLoader

    paths = [os.path.join(os.environ['PROBE_SOURCE'], 'tests/hsuite/fixture_na/WFN.h5'),
             '/pscratch/sd/j/jackm/cs_claude/mos2v2/runs/c9_srlr/WFN.h5']

    def snapshot(wfn):
        return tuple({key:id(value) for key,value in getattr(wfn, name).items()}
                     for name in ('_gvecs_cache', '_gvecs_dev_cache', '_ngk_valid_cache'))

    for path in paths:
        with WfnLoader(path, mesh=stack.mesh) as wfn, h5py.File(path) as file:
            grid = np.asarray(wfn.fft_grid, dtype=np.int64)
            nr = int(np.prod(grid))
            before = snapshot(wfn)
            for first in range(int(wfn.nkpts)):
                rows = [first, (first+1) % int(wfn.nkpts), first]
                request = IBZRows(rows)
                vectors = wfn.gvecs(k=request, cache=False)
                lengths = wfn.ngk_valid(k=request, cache=False)
                host = wfn.box_index(k=request, cache=False)
                device = wfn.box_index_dev(k=request, mesh=stack.mesh, cache=False)
                expected = np.broadcast_to(nr+np.arange(int(wfn.ngkmax)), host.shape).copy()
                for pos, k in enumerate(rows):
                    ng = int(wfn.ngk[k])
                    start = int(wfn.kpt_starts[k])
                    raw = np.asarray(file['wfns/gvecs'][start:start+ng], dtype=np.int32)
                    np.testing.assert_array_equal(vectors[pos, :ng], raw)
                    assert lengths[pos] == ng
                    # Literal C-order FFT-box coordinates, independent of the
                    # production sphere-index helper and its padding logic.
                    cell = np.mod(raw.astype(np.int64), grid)
                    expected[pos, :ng] = ((cell[:, 0]*grid[1]+cell[:, 1])*grid[2]+cell[:, 2])
                np.testing.assert_array_equal(host, expected)
                for shard in device.addressable_shards:
                    np.testing.assert_array_equal(np.asarray(shard.data), expected[shard.index])
                assert snapshot(wfn) == before, 'transient request retained a nested table'
                device.delete()
            request = IBZRows([0, min(1, int(wfn.nkpts)-1)])
            host = wfn.box_index(k=request)
            assert wfn.box_index(k=request) is host
            device = wfn.box_index_dev(k=request, mesh=stack.mesh)
            assert wfn.box_index_dev(k=request, mesh=stack.mesh) is device
            before = snapshot(wfn)
            transient = wfn.box_index_dev(k=request, mesh=stack.mesh, cache=False)
            np.testing.assert_array_equal(np.asarray(transient.addressable_shards[0].data), host)
            assert snapshot(wfn) == before
            transient.delete()
            if jax.process_index() == 0:
                print(f'TRANSIENT_WFN_INDEX PASS path={path} requests={int(wfn.nkpts)}', flush=True)
    runtime.finalize_process(0)


def test_requires_distributed_launcher():
    import pytest
    pytest.skip('run directly under the configured MPI launcher')


if __name__ == '__main__':
    main()

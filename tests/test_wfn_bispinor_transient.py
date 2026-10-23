"""Bispinor subset/metric streams retain no k-set G tables; literal Pauli lift."""


def main():
    import runtime
    stack = runtime.initialize_communicator_stack()
    import json
    from pathlib import Path
    import h5py
    import jax
    import numpy as np
    from centroid.sampling_metric import build_feature_metric_diagonal
    from wfn_loader import IBZRows, WfnLoader

    def inventory(wfn):
        return tuple({key:id(value) for key,value in getattr(wfn, name).items()}
                     for name in ('_gvecs_cache', '_gvecs_dev_cache', '_ngk_valid_cache'))

    fixture = Path(__file__).parent/'hsuite/fixture/WFN.h5'
    with WfnLoader(fixture, mesh=stack.mesh, backend='phdf5') as wfn:
        before = inventory(wfn)
        worst = 0.
        for rows in ((0, 1, 0), (int(wfn.nkpts)-1, 2)):
            request = IBZRows(rows)
            for lift in ('raw', 'normalized_rkb'):
                value = wfn.load(bands=(1, 4), k=request, bispinor=True,
                                 bispinor_lift=lift)
                value.block_until_ready()
                wfn.sync_read_staging()
                expected = np.zeros(value.shape, dtype=np.complex128)
                with h5py.File(fixture) as raw:
                    for out, k in enumerate(rows):
                        start = int(wfn.kpt_starts[k]); ng = int(wfn.ngk[k])
                        packed = raw['wfns/coeffs'][1:4, :, start:start+ng, :]
                        large = packed[..., 0]+1j*packed[..., 1]
                        g = raw['wfns/gvecs'][start:start+ng]
                        cart = (g+np.asarray(wfn.kpoints[k]))@(float(wfn.blat)*np.asarray(wfn.bvec))
                        x,y,z = cart.T
                        up,down = large[:, 0],large[:, 1]
                        small = .00364867628215*np.stack((
                            z*up+(x-1j*y)*down, (x+1j*y)*up-z*down), axis=1)
                        lifted = np.concatenate((large, small), axis=1)
                        if lift == 'normalized_rkb':
                            lifted /= np.sqrt(1+.00364867628215**2*np.sum(cart**2,axis=1))[None,None,:]
                        expected[out, :3, :, :ng] = lifted
                for shard in value.addressable_shards:
                    actual = np.asarray(shard.data); ref = expected[shard.index]
                    error = float(np.max(np.abs(actual-ref), initial=0))/max(1e-14,float(np.max(np.abs(ref), initial=0)))
                    worst = max(worst,error)
                    assert error < 2e-13, (rows, lift, error)
                    assert np.all(actual[ref == 0] == 0)
                value.delete()
                assert inventory(wfn) == before, '4c subset retained a G/index table'
        full_rows = (int(wfn.symmetry().nk_tot)-1, 0, 2)
        value = wfn.load(bands=(1, 4), k=full_rows, bispinor=True)
        value.block_until_ready(); value.delete()
        assert inventory(wfn) == before, '4c full-zone subset retained a G/index table'
        metric = build_feature_metric_diagonal(wfn, wfn.symmetry(), (0, 2), (1, 3),
                    gamma_mode='transverse', dist_mesh=stack.mesh, verbose=False)
        assert np.all(np.isfinite(metric)) and np.min(metric) >= 0
        assert inventory(wfn) == before, 'transverse metric retained a G/index table'
        print('BISPINOR_TRANSIENT '+json.dumps(dict(status='PASS',rank=jax.process_index(),
            max_literal_relative=worst, metric_max=float(np.max(metric)))),flush=True)
    runtime.finalize_process(0)


def test_requires_distributed_launcher():
    import pytest
    pytest.skip('run directly under the configured MPI launcher')


if __name__ == '__main__':
    main()

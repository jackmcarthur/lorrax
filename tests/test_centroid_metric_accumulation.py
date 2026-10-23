"""Metric accumulator donation preserves values and the undonated inputs."""


def main():
    import runtime
    stack = runtime.initialize_communicator_stack()
    import json
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import barrier, device_put_process_local
    from centroid.sampling_metric import (
        _accumulate_density_matrix, _accumulate_grid_pullback,
        _accumulate_metric_rows)

    sharding=NamedSharding(stack.mesh,P())
    def put(value):
        return device_put_process_local(np.asarray(value),sharding)
    def ptr(value):
        return value.unsafe_buffer_pointer()
    ownership = []
    rng=np.random.default_rng(7314)
    grid=(2,3,5)
    left=rng.normal(size=(2,2,2,*grid))+1j*rng.normal(size=(2,2,2,*grid))
    right=rng.normal(size=left.shape)+1j*rng.normal(size=left.shape)
    weights=np.asarray([[.25,.5,.75],[1.,.125,.25]])
    field=np.real(np.sum(left*np.swapaxes(right,1,2),axis=(1,2)))
    expected_rows=np.einsum('kg,kr->gr',weights,field.reshape(2,-1))
    initial=put(np.zeros((3,30),np.float64)); before=ptr(initial)
    dl,dr=put(left),put(right)
    rows=_accumulate_metric_rows(initial,dl,dr,put(weights),mode='charge')
    rows.block_until_ready()
    np.testing.assert_allclose(np.asarray(rows),expected_rows,rtol=2e-13,atol=2e-13)
    ownership.append(dict(kernel='rows', old_deleted=initial.is_deleted(),
                          pointer_reused=ptr(rows)==before))
    assert not dl.is_deleted() and not dr.is_deleted()

    old=put(np.zeros(30,np.float64)); before=ptr(old)
    physical=np.arange(30,dtype=np.float64)/16
    source=np.arange(29,-1,-1,dtype=np.int32)
    physical_dev=put(physical)
    result=_accumulate_grid_pullback(old,physical_dev,put(source),.25)
    result.block_until_ready()
    np.testing.assert_array_equal(np.asarray(result),.25*physical[source])
    ownership.append(dict(kernel='grid', old_deleted=old.is_deleted(),
                          pointer_reused=ptr(result)==before))
    assert not physical_dev.is_deleted()

    # On the first band tile the part IS the running density. Releasing its
    # Python name is safe; the next addition donates only the old running one.
    first=rng.normal(size=(2,2,3))+1j*rng.normal(size=(2,2,3))
    second=rng.normal(size=first.shape)+1j*rng.normal(size=first.shape)
    part=put(first); running=part; del part
    second_dev=put(second); before=ptr(running)
    density=_accumulate_density_matrix(running,second_dev)
    density.block_until_ready()
    np.testing.assert_array_equal(np.asarray(density),first+second)
    ownership.append(dict(kernel='density', old_deleted=running.is_deleted(),
                          pointer_reused=ptr(density)==before))
    assert not second_dev.is_deleted()
    print('METRIC_ACCUMULATION '+json.dumps(dict(status='PASS',rank=jax.process_index(),
        other_inputs_valid=True,first_alias_valid=True,ownership=ownership)),flush=True)
    barrier('metric-accumulation-proof')
    runtime.finalize_process(0)


def test_requires_distributed_launcher():
    import pytest
    pytest.skip('run directly under the configured MPI launcher')


if __name__=='__main__':
    main()

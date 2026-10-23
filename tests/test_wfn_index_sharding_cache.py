"""Index cache keys preserve both numeric values and requested ownership."""


def main():
    import runtime
    stack = runtime.initialize_communicator_stack()
    from pathlib import Path
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from wfn_loader import IBZRows, WfnLoader

    fixture = Path(__file__).parent/'hsuite/fixture/WFN.h5'
    with WfnLoader(fixture, mesh=stack.mesh) as wfn:
        requests = (IBZRows((0, 1, 2, 3)), IBZRows((3, 2, 1, 0)))
        for request, specs in zip(requests, ((P(None,None),P('x',None),P('y',None)),
                                           (P('y',None),P('x',None),P(None,None)))):
            expected = wfn.box_index(k=request)
            outputs = []
            for spec in specs:
                placement = NamedSharding(stack.mesh,spec)
                value = wfn.box_index_dev(k=request,mesh=stack.mesh,sharding=spec)
                assert value.sharding.is_equivalent_to(placement,ndim=2), (spec,value.sharding)
                for shard in value.addressable_shards:
                    np.testing.assert_array_equal(np.asarray(shard.data),expected[shard.index])
                assert wfn.box_index_dev(k=request,mesh=stack.mesh,sharding=placement) is value
                outputs.append(value)
            assert len({id(value) for value in outputs}) == 3
            default = wfn.box_index_dev(k=request,mesh=stack.mesh)
            assert default is outputs[specs.index(P(None,None))]
            default_key = ('box_index_dev',*wfn._k_cache_key(request),
                           tuple(int(s) for s in wfn.fft_grid),id(stack.mesh))
            assert wfn._gvecs_dev_cache[default_key] is default
            before = {key:id(value) for key,value in wfn._gvecs_dev_cache.items()}
            transient = wfn.box_index_dev(k=request,mesh=stack.mesh,sharding=P('x',None),cache=False)
            assert transient is not outputs[specs.index(P('x',None))]
            assert transient.sharding.is_equivalent_to(NamedSharding(stack.mesh,P('x',None)),ndim=2)
            assert {key:id(value) for key,value in wfn._gvecs_dev_cache.items()} == before
            transient.delete()
    print(f'INDEX_SHARDING_CACHE PASS rank={jax.process_index()}',flush=True)
    runtime.finalize_process(0)


def test_requires_distributed_launcher():
    import pytest
    pytest.skip('run directly under the configured MPI launcher')


if __name__ == '__main__':
    main()

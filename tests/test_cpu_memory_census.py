"""Real CPU buffer aliases must not reduce admission room twice or form cycles."""
import gc
import json
import time
import weakref


def main():
    import runtime
    runtime.bootstrap()
    stack=runtime.initialize_communicator_stack()
    import jax
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from lxkit import memory

    assert jax.local_devices()[0].platform=='cpu'

    def resident():
        return int(memory._in_use(per_device=False))

    def local_alias_and_distinct_allocations():
        baseline=resident()
        a=jax.device_put(np.ones(1024,np.float64)); a.block_until_ready()
        assert resident()-baseline==a.nbytes
        local=a.addressable_data(0)
        assert a.unsafe_buffer_pointer()==local.unsafe_buffer_pointer()
        assert resident()-baseline==a.nbytes
        b=jax.device_put(np.full(1024,2.,np.float64)); b.block_until_ready()
        assert a.unsafe_buffer_pointer()!=b.unsafe_buffer_pointer()
        assert resident()-baseline==a.nbytes+b.nbytes
        del a
        assert resident()-baseline==local.nbytes+b.nbytes
        del local,b
        gc.collect()
        assert resident()==baseline

    def overlapping_external_views():
        baseline=resident()
        raw=np.empty(2056,np.float64)
        offset=(-raw.ctypes.data//raw.itemsize)%8
        aligned=raw[offset:offset+2048]
        a=jax.dlpack.from_dlpack(aligned[:1024],copy=False)
        b=jax.dlpack.from_dlpack(aligned[512:1536],copy=False)
        assert a.unsafe_buffer_pointer()==aligned.ctypes.data
        assert b.unsafe_buffer_pointer()==aligned.ctypes.data+512*8
        assert resident()-baseline==1536*8
        del a
        assert resident()-baseline==1024*8
        del b
        gc.collect()
        assert resident()==baseline

    def single_device_named_sharding():
        baseline=resident()
        mesh=Mesh(np.array([jax.local_devices()[0]],dtype=object),('local',))
        sharding=NamedSharding(mesh,P())
        a=jax.device_put(np.ones(1024,np.float64),sharding)
        a.block_until_ready()
        assert isinstance(a.sharding,NamedSharding) and len(a.devices())==1
        # On JAX, a.device is this NamedSharding rather than a Device,
        # even though there is exactly one device holding the buffer.
        assert resident()-baseline==a.nbytes
        local=a.addressable_data(0)
        assert resident()-baseline==a.nbytes
        del local,a
        gc.collect()
        assert resident()==baseline

    def distributed_array_and_local_alias():
        baseline=resident()
        shape=(16*jax.process_count(),16)
        sharding=NamedSharding(stack.mesh,P('x','y'))
        a=jax.make_array_from_callback(shape,sharding,
            lambda ix:np.ones(tuple(len(range(*s.indices(n))) for s,n in zip(ix,shape)),np.float64))
        a.block_until_ready()
        nbytes=int(np.prod(sharding.shard_shape(shape))*8)
        assert resident()-baseline==nbytes
        local=a.addressable_data(0)
        assert resident()-baseline==nbytes
        del local,a
        gc.collect()
        assert resident()==baseline

    def observer_does_not_form_cycles():
        a=jax.device_put(np.ones(1024,np.float64)); a.block_until_ready()
        local=a.addressable_data(0)
        ref=weakref.ref(local)
        del a
        gc.collect()
        was_enabled=gc.isenabled()
        gc.disable()
        resident()
        del local
        alive=ref() is not None
        if was_enabled: gc.enable()
        assert not alive,'memory reader retained a local shard until cyclic GC'

    def replicated_mesh_array_and_local_alias():
        baseline=resident()
        sharding=NamedSharding(stack.mesh,P())
        a=jax.device_put(np.ones((12,6),np.complex128),sharding)
        a.block_until_ready()
        expected=a.nbytes*len(sharding.addressable_devices)
        assert resident()-baseline==expected
        local=a.addressable_data(0)
        assert resident()-baseline==expected
        del a
        assert resident()-baseline==local.nbytes
        del local
        gc.collect()
        assert resident()==baseline

    for case in (local_alias_and_distinct_allocations,overlapping_external_views,single_device_named_sharding,
                 distributed_array_and_local_alias,observer_does_not_form_cycles,
                 replicated_mesh_array_and_local_alias):
        case()
    t=time.perf_counter()
    for _ in range(100): resident()
    print('CPU_CENSUS PASS '+json.dumps(dict(rank=jax.process_index(),cases=6,
          empty_read_mean_seconds=(time.perf_counter()-t)/100)),flush=True)
    runtime.finalize_process(0)


if __name__=='__main__':
    main()

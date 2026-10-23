"""P4 GPU alias proof for eager Cq/factor/face boundaries.

An independent eager-constraint control must allocate one complete face;
production matching boundaries must preserve pointers and active/dummy values.
Genuine resharding and traced constraints are also exercised.
Run as a compute-node script with --out; no physics or physical-capacity claim.
"""

def main():
    import argparse
    import ctypes
    import gc
    import json
    from pathlib import Path
    from types import SimpleNamespace

    import runtime
    mesh = runtime.initialize_communicator_stack().mesh
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from distrib_la import ensure_sharding
    from isdf.core import _identity_pad_block_diagonal
    from common.wfn_transforms import _centroid_face_kernels

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert mesh.size == 4 and len(jax.local_devices()) == 1
    face = NamedSharding(mesh, P(None, 'x', 'y'))
    lib = ctypes.CDLL('libcuda.so.1')

    def pointer(a):
        return int(a.addressable_data(0).unsafe_buffer_pointer())

    pool = ctypes.c_void_p()
    def current():
        assert lib.cuCtxSynchronize() == 0
        value = ctypes.c_uint64()
        assert lib.cuMemPoolGetAttribute(pool, 7, ctypes.byref(value)) == 0
        return int(value.value)

    def values(shape, index):
        q, r, c = [np.arange(*s.indices(n)) for s, n in zip(index, shape)]
        z = q[:,None,None] + .01*r[None,:,None] + .00001*c[None,None,:]
        z = z + 1j*(.001*q[:,None,None] - .01*r[None,:,None] + .0001*c[None,None,:])
        return np.where((r[None,:,None] < shape[1]-3) & (c[None,None,:] < shape[2]-3), z, 0)

    shape = (4, 1024, 1024)
    x = jax.make_array_from_callback(shape, face, lambda i: values(shape, i))
    x.block_until_ready()
    assert lib.cuPointerGetAttribute(ctypes.byref(pool), 17, ctypes.c_uint64(pointer(x))) == 0
    assert pool.value
    old = lambda a: jax.lax.with_sharding_constraint(a.reshape(shape), face)
    warm = old(x);warm.block_until_ready();del warm;gc.collect();current()
    base = current()
    y = old(x);y.block_until_ready()
    old_delta = current()-base
    assert pointer(y) != pointer(x) and old_delta >= x.addressable_data(0).nbytes
    np.testing.assert_array_equal(np.asarray(y.addressable_data(0)), np.asarray(x.addressable_data(0)))
    old_ptr = pointer(y)
    del y;gc.collect();current()
    base = current()
    z = ensure_sharding(x.reshape(shape), face)
    assert z is x and pointer(z) == pointer(x) and current() == base
    other = NamedSharding(mesh, P('x', None, 'y'))
    changed = ensure_sharding(x, other);changed.block_until_ready()
    assert changed.sharding.spec == other.spec and pointer(changed) != pointer(x)
    idx = other.devices_indices_map(shape)[jax.local_devices()[0]]
    np.testing.assert_array_equal(np.asarray(changed.addressable_data(0)), values(shape, idx))

    trace = jax.make_jaxpr(lambda a: ensure_sharding(a, face))(
        jax.ShapeDtypeStruct(shape, jnp.complex128, sharding=face))
    assert any(e.primitive.name == 'sharding_constraint' for e in trace.jaxpr.eqns)
    exe = jax.jit(lambda a: ensure_sharding(a, face), in_shardings=face,
                  out_shardings=face).lower(x).compile()
    m = exe.memory_analysis()
    equal = exe(x);equal.block_until_ready()
    np.testing.assert_array_equal(np.asarray(equal.addressable_data(0)), np.asarray(x.addressable_data(0)))

    no_pad = _identity_pad_block_diagonal(x, n_rmu_logical=shape[1], mesh_xy=mesh)
    no_pad.block_until_ready()
    assert (pointer(no_pad) == pointer(x)) == True
    np.testing.assert_array_equal(np.asarray(no_pad.addressable_data(0)), np.asarray(x.addressable_data(0)))
    # The real padded branch must still insert identity only in the dummy block.
    padded = _identity_pad_block_diagonal(x, n_rmu_logical=shape[1]-3, mesh_xy=mesh)
    padded.block_until_ready()
    local_index = face.devices_indices_map(shape)[jax.local_devices()[0]]
    q, r, c = [np.arange(*s.indices(n)) for s, n in zip(local_index, shape)]
    want = values(shape, local_index) + ((r[:,None] == c[None,:]) & (r[:,None] >= shape[1]-3))[None]
    np.testing.assert_array_equal(np.asarray(padded.addressable_data(0)), want)

    # Existing face finalizer, no slice/no user pad: both input layouts already match.
    sy = NamedSharding(mesh, P(None, None, None, 'y'))
    sx = NamedSharding(mesh, P(None, 'x', None, None))
    ay = jax.device_put(np.arange(2*8*2*10).reshape(2,8,2,10).astype(complex), sy)
    ax = jax.device_put(np.conj(np.arange(2*8*2*10).reshape(2,8,2,10).transpose(0,3,1,2)).astype(complex), sx)
    _, finish = _centroid_face_kernels(0, SimpleNamespace(b_id_4_user=8), None,
        10, 10, 8, sx, sy, sx, sy)
    by, bx = finish(ay, ax, 2)
    jax.block_until_ready((by,bx))
    assert (pointer(by) == pointer(ay)) == True
    assert (pointer(bx) == pointer(ax)) == True
    np.testing.assert_array_equal(np.asarray(by.addressable_data(0)), np.asarray(ay.addressable_data(0)))
    np.testing.assert_array_equal(np.asarray(bx.addressable_data(0)), np.asarray(ax.addressable_data(0)))

    out = {'rank':jax.process_index(), 'candidate':True,
           'payload_per_rank':int(x.addressable_data(0).nbytes),
           'eager_constraint_delta':old_delta, 'ensure_matching_delta':0,
           'input_pointer':pointer(x), 'old_constraint_pointer':old_ptr,
           'ensure_pointer':pointer(z), 'no_pad_pointer':pointer(no_pad),
           'genuine_reshard_pointer':pointer(changed),
           'face_finish_input_pointers':[pointer(ay),pointer(ax)],
           'face_finish_output_pointers':[pointer(by),pointer(bx)],
           'traced_constraint':True, 'aot':{k:int(getattr(m,k)) for k in
             ['argument_size_in_bytes','output_size_in_bytes','alias_size_in_bytes','temp_size_in_bytes']},
           'dummy_and_active_values':'exact', 'verdict':'PASS'}
    args.out.mkdir(exist_ok=True,parents=True)
    (args.out/f'rank{jax.process_index()}.json').write_text(json.dumps(out,indent=2)+'\n')
    print('EAGER_SHARDING PASS',jax.process_index(),flush=True)


if __name__ == '__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)

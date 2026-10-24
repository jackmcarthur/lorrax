"""Owned ladder kernels release W_q without changing the real-space operator."""
import jax
import numpy as np
from jax.sharding import Mesh


def check_kernel_residency(mesh, n_mu=8, grid=(3,3,1)):
    import jax.numpy as jnp
    from bse.bse_feast import ensure_W_R
    from bse.bse_ring_comm import make_bse_shardings
    from bse.head_resolvent import zero_kernel_payload
    from common.collectives import device_put_process_local
    from common.fft_helpers import get_donated_kfft_kminor

    jax.config.update('jax_enable_x64',True)
    shape=(n_mu,n_mu,*grid)
    raw=np.arange(np.prod(shape),dtype=np.float64).reshape(shape)/np.prod(shape)
    raw=raw.astype(np.complex128)+1j*np.sin(raw)
    expected=np.fft.ifftn(raw,axes=(-3,-2,-1),norm='ortho')
    sh=make_bse_shardings(mesh)
    put=lambda value,layout:device_put_process_local(np.array(value,copy=True),layout)
    borrowed=dict(W_q=put(raw,sh.W),V_q0=put(np.eye(n_mu),sh.V))
    snapshot=borrowed['W_q']
    ensure_W_R(borrowed,True,mesh)
    jax.block_until_ready(borrowed['W_R'])
    assert borrowed['W_q'] is snapshot and not snapshot.is_deleted()
    for a in borrowed['W_R'].addressable_shards:
        np.testing.assert_allclose(a.data,expected[a.index],rtol=3e-12,atol=3e-12)
    for a in snapshot.addressable_shards:
        np.testing.assert_array_equal(a.data,raw[a.index])

    allocated_before=jax.local_devices()[0].memory_stats()
    owned=dict(W_q=put(raw,sh.W),V_q0=put(np.eye(n_mu),sh.V))
    original=owned['W_q']
    jax.block_until_ready(original)
    original_pointers=tuple(s.data.unsafe_buffer_pointer() for s in original.addressable_shards)
    transform=get_donated_kfft_kminor(mesh,grid,sh.W.spec,kind='ifftn',norm='ortho')
    analysis=transform.lower(original).compile().memory_analysis()
    ensure_W_R(owned,True,mesh,consume_W_q=True)
    jax.block_until_ready((owned['W_R'],owned['_W_q0_for_precond']))
    assert owned['W_q'] is None and original.is_deleted()
    assert analysis.alias_size_in_bytes>0
    pointer_reused=original_pointers==tuple(s.data.unsafe_buffer_pointer() for s in owned['W_R'].addressable_shards)
    # The CPU runtime may copy a donated allocation at its dispatch boundary.
    # The native GPU control additionally measures actual in-place residency.
    if jax.local_devices()[0].platform!='cpu':
        assert pointer_reused
    for a in owned['W_R'].addressable_shards:
        np.testing.assert_allclose(a.data,expected[a.index],rtol=3e-12,atol=3e-12)
    for a in owned['_W_q0_for_precond'].addressable_shards:
        np.testing.assert_array_equal(a.data,raw[:,:,0,0,0][a.index])
    allocated_after=jax.local_devices()[0].memory_stats()
    prepared=owned['W_R']
    # A body and its later Gamma head may borrow the same prepared operand.
    for include_w in (True,False,True):
        ensure_W_R(owned,include_w,mesh)
        assert owned['W_R'] is prepared
    free=zero_kernel_payload(owned)
    ensure_W_R(free,True,mesh)
    for name in ('W_R','V_q0','_W_q0_for_precond'):
        for a in free[name].addressable_shards:
            np.testing.assert_array_equal(a.data,0)
    assert owned['W_R'] is prepared

    def local_bytes(data):
        return sum(a.addressable_shards[0].data.nbytes for name in ('W_q','W_R','_W_q0_for_precond')
                   if (a:=data.get(name)) is not None)
    before,after=local_bytes(borrowed),local_bytes(owned)
    expected_saved=(np.prod(shape)-n_mu*n_mu)*16//mesh.size
    assert before-after==expected_saved
    return dict(borrowed_kernel_bytes_per_rank=before,owned_kernel_bytes_per_rank=after,
                saved_bytes_per_rank=before-after,alias_bytes_per_rank=analysis.alias_size_in_bytes,
                input_pointer_reused=pointer_reused,
                allocator_before_owned=allocated_before,allocator_after_owned=allocated_after)


def test_owned_kernel_preserves_values_borrowers_and_preconditioner():
    from ffi.fft import xla_reference
    from lxkit import memory
    memory.set_budget(2**30,platform='cpu')
    with xla_reference():
        for side in ((1,2) if len(jax.devices('cpu'))>=4 else (1,)):
            mesh=Mesh(np.asarray(jax.devices('cpu')[:side*side]).reshape(side,side),('x','y'))
            for grid in ((1,1,1),(3,3,1),(2,3,2)):
                check_kernel_residency(mesh,grid=grid)

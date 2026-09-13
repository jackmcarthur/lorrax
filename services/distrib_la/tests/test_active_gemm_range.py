"""Exact contiguous contraction over fixed distributed GEMM allocations."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from common.collectives import device_put_process_local
from distrib_la import gemm_plan


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('dtype,beta', [(np.float64, 0.), (np.complex128, -.3+.1j)])
def test_active_range_matches_dense_without_reading_inactive_bands(dtype, beta, batched):
    if jax.process_count() != 4:
        pytest.skip('Requires four processes and four GPUs')
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ('x', 'y'))
    sharding = NamedSharding(mesh, P(None, 'x', 'y'))
    replicated = NamedSharding(mesh, P())
    nq, m, k, n = 3, 32, 64, 48
    rng = np.random.default_rng(219)
    def values(shape):
        a = rng.standard_normal(shape)
        if np.issubdtype(dtype, np.complexfloating):
            a = a + 1j*rng.standard_normal(shape)
        return a.astype(dtype)
    a, b, c = values((nq,m,k)), values((nq,k,n)), values((nq,m,n))
    alpha = .7+.2j if np.issubdtype(dtype,np.complexfloating) else .7
    plan = gemm_plan(mesh,m=m,k=k,n=n,nq=nq,dtype=jnp.dtype(dtype),
                     alpha=alpha,beta=beta,enable_active_range=True)
    def put(v, sh=sharding):
        return device_put_process_local(v, sh)
    if beta == 0:
        fn = jax.jit(lambda aa,bb,lo,hi,cc:plan.active_range(aa,bb,lo,hi))
    else:
        fn = jax.jit(lambda aa,bb,lo,hi,cc:plan.active_range(aa,bb,lo,hi,C=cc))
    initial_lo=np.zeros(nq,np.int32) if batched else np.int32(0)
    initial_hi=np.full(nq,k,np.int32) if batched else np.int32(k)
    args=(put(a),put(b),put(initial_lo,replicated),put(initial_hi,replicated),put(c))
    executable=fn.lower(*args).compile()
    for lo,hi in ((0,k),(5,11),(30,38),(32,64),(0,0),(7,7)):
        aa,bb=a.copy(),b.copy()
        # Inactive inputs are deliberately NaN. Masked full-K BLAS cannot
        # pass this gate; descriptor views must never touch these bands.
        los=np.asarray([lo,0,37],np.int32) if batched else np.full(nq,lo,np.int32)
        his=np.asarray([hi,k,37],np.int32) if batched else np.full(nq,hi,np.int32)
        expected=np.empty_like(c)
        for batch,(begin,end) in enumerate(zip(los,his)):
            aa[batch,:,:begin]=np.nan;aa[batch,:,end:]=np.nan
            bb[batch,:begin,:]=np.nan;bb[batch,end:,:]=np.nan
            expected[batch]=alpha*(a[batch,:,begin:end]@b[batch,begin:end,:])+beta*c[batch]
        actual=executable(put(aa),put(bb),put(los if batched else np.int32(lo),replicated),
                          put(his if batched else np.int32(hi),replicated),put(c))
        assert tuple(actual.sharding.spec)==(None,'x','y')
        for shard in actual.addressable_shards:
            np.testing.assert_allclose(np.asarray(shard.data),expected[shard.index],
                                       rtol=2e-12,atol=2e-12)
            assert shard.data.size*4==actual.size


@pytest.mark.parametrize('dtype', [np.float64,np.complex128])
def test_active_full_range_is_bitwise_original_dense(dtype):
    if jax.process_count()!=4:
        pytest.skip('Requires four processes and four GPUs')
    mesh=Mesh(np.asarray(jax.devices()).reshape(2,2),('x','y'))
    sh=NamedSharding(mesh,P(None,'x','y'))
    rng=np.random.default_rng(124)
    a=rng.normal(size=(3,32,64)).astype(dtype)
    b=rng.normal(size=(3,64,48)).astype(dtype)
    if np.issubdtype(dtype,np.complexfloating):
        a+=1j*rng.normal(size=a.shape);b+=1j*rng.normal(size=b.shape)
    a=device_put_process_local(a,sh);b=device_put_process_local(b,sh)
    plan=gemm_plan(mesh,m=32,k=64,n=48,nq=3,dtype=jnp.dtype(dtype),enable_active_range=True)
    dense=plan(a,b)
    for lo,hi in ((0,64),(jnp.zeros(3,jnp.int32),jnp.full(3,64,jnp.int32))):
        active=plan.active_range(a,b,lo,hi)
        for expected,actual in zip(dense.addressable_shards,active.addressable_shards):
            np.testing.assert_array_equal(np.asarray(expected.data),np.asarray(actual.data))

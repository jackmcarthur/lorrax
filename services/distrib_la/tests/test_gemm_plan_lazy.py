"""A plan without dummy warmup works on its first nested execution."""
import importlib
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P
from common.collectives import device_put_process_local
from distrib_la import gemm_plan

@pytest.mark.parametrize('layout', ['face','axis'])
@pytest.mark.parametrize('active', [False,True])
@pytest.mark.parametrize('beta', [0.,.3])
def test_no_dummy_allocations_first_scan_and_donation(monkeypatch,layout,active,beta):
    cpu=jax.devices()[0].platform=='cpu'
    if jax.device_count()!=4 or (layout=='face' and jax.process_count()!=4 and not cpu):
        pytest.skip('requires P4 CUDA or a four-device CPU mesh')
    mesh=Mesh(np.asarray(jax.devices()).reshape(2,2),('x','y'))
    module=importlib.import_module('distrib_la.matmul_plan')
    def forbidden(*args,**kwargs):
        raise AssertionError('lazy planning allocated dummy operands')
    with monkeypatch.context() as m:
        m.setattr(module,'_zeros',forbidden)
        plan=gemm_plan(mesh,m=12,k=16,n=8,nq=3,dtype=jnp.complex128,
                       layout=layout,beta=beta,enable_active_range=active,warmup=False)
    rng=np.random.default_rng(418)
    def values(shape):return rng.normal(size=shape)+1j*rng.normal(size=shape)
    a,b,c=values((3,12,16)),values((3,16,8)),values((3,12,8))
    lo,hi=(3,11) if active else (0,16)
    product=a[:,:,lo:hi]@b[:,lo:hi,:]
    expected=c.copy()
    for _ in range(3):expected=product+beta*expected
    aa=device_put_process_local(a,plan.in_sharding_a)
    bb=device_put_process_local(b,plan.in_sharding_b)
    cc=device_put_process_local(c,plan.out_sharding)
    @partial(jax.jit, donate_argnums=(2,))
    def fn(aa,bb,cc):
        def step(cc,_):
            if active:
                out=plan.active_range(aa,bb,lo,hi,C=cc) if beta else plan.active_range(aa,bb,lo,hi)
            else:
                out=plan(aa,bb,C=cc) if beta else plan(aa,bb)
            return out,None
        return jax.lax.scan(step,cc,None,length=3)[0]
    result=fn(aa,bb,cc);result.block_until_ready()
    assert tuple(result.sharding.spec)==(None,'x','y')
    for shard in result.addressable_shards:
        assert shard.data.size*4==result.size
        np.testing.assert_allclose(np.asarray(shard.data),expected[shard.index],atol=2e-12,rtol=2e-12)

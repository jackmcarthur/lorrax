"""FACE cross-Grams use the shared two-axis GEMM with complex weights and tails."""
import re

import numpy as np
import pytest


@pytest.mark.parametrize('side', [2, 3])
def test_face_cross_grams_and_single_shells_use_owned_matrix_products(monkeypatch, side):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from isdf import zeta_mubatch as zm
    import distrib_la
    jax.config.update('jax_enable_x64', True)
    if len(jax.devices('cpu')) < side*side:
        pytest.skip('requires nine virtual CPU devices')
    mesh = Mesh(np.array(jax.devices('cpu')[:side*side]).reshape(side, side), ('x','y'))
    q, logical, mu, tile = side*side+1, 13, -(-13//(side*side))*side*side, 4*side
    rng = np.random.default_rng(990731+side)
    raw = rng.normal(size=(3,q,mu,2*tile))+1j*rng.normal(size=(3,q,mu,2*tile))
    raw[:,:,logical:]=0
    ngk = 2*tile-1-np.arange(q)%3
    valid = np.arange(2*tile)[None,:] < ngk[:,None]
    expected_z = np.where(valid[None,:,None,:],raw,0)
    raw = np.where(valid[None,:,None,:],raw,np.nan+1j*np.nan)
    weights = rng.normal(size=(6,q,2*tile))+1j*rng.normal(size=(6,q,2*tile))
    weights = np.where(valid[None,:,:],weights,np.nan+1j*np.nan)
    expected_w = np.where(valid[None,:,:],weights,0)
    slots = np.stack([np.zeros(q,np.int32),ngk-1],axis=1)
    pairs = ((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))
    face = NamedSharding(mesh,P(None,'x','y'))
    mush = NamedSharding(mesh,P(None,('x','y'),None))
    put = lambda a, sh:jax.device_put(a,sh)
    count=[];original=distrib_la.matmul
    def product(*args,**kwargs):
        count.append((kwargs['transb'],kwargs['batched_route'],kwargs.get('budget_bytes')))
        return original(*args,**kwargs)
    monkeypatch.setattr(distrib_la,'matmul',product)
    step=zm._face_pair_kernel(mesh,tile,pairs,3)
    V=tuple(put(np.zeros((q,mu,mu),complex),face) for _ in pairs)
    S=tuple(put(np.zeros((q,mu,2),complex),mush) for _ in range(3))
    w,ng,sl=jnp.asarray(weights),jnp.asarray(ngk,jnp.int32),jnp.asarray(slots,jnp.int32)
    for t in range(2):
        z=tuple(put(row[:,:,(t*tile):((t+1)*tile)],face) for row in raw)
        if t==0:
            executable=step.lower(z,w,ng,sl,jnp.int32(t),V,S).compile()
            hlo=executable.as_text()
            # The former implicit einsum made this one-axis partial V on CPU0.12.
            assert not re.search(r'c128\['+f'{q},{mu//side},{mu}'+r'\]',hlo)
        V,S=executable(z,w,ng,sl,jnp.int32(t),V,S)
    jax.block_until_ready((V,S))
    assert count and all(x==('T','auto',None) for x in count)
    for i,(a,b) in enumerate(pairs):
        expected=np.einsum('qmg,qg,qng->qmn',expected_z[a].conj(),expected_w[i],expected_z[b])
        np.testing.assert_allclose(np.asarray(V[i]),expected,rtol=2e-12,atol=3e-13)
    for i in range(3):
        expected=np.take_along_axis(expected_z[i],slots[:,None,:],axis=2)
        np.testing.assert_allclose(np.asarray(S[i]),expected,rtol=2e-12,atol=3e-14)
    # Exercise the distinct public single-FACE accumulator with the same literal equation.
    _,_,acc=zm._face_kernels(mesh,q,tile,True)
    v=put(np.zeros((q,mu,mu),complex),face)
    shell=put(np.zeros((q,mu,2),complex),mush)
    for t in range(2):
        v,shell,_=acc(put(raw[0,:,:,t*tile:(t+1)*tile],face),w[0],ng,sl,jnp.int32(t),v,shell)
    jax.block_until_ready((v,shell))
    np.testing.assert_allclose(np.asarray(v),np.einsum('qmg,qg,qng->qmn',
        expected_z[0].conj(),expected_w[0],expected_z[0]),rtol=2e-12,atol=3e-13)
    np.testing.assert_allclose(np.asarray(shell),np.take_along_axis(
        expected_z[0],slots[:,None,:],axis=2),rtol=2e-12,atol=3e-14)

"""Runtime eigenpair conditioning; CPU4 is separate from actual MPI4 proof."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from isdf import cplus


@pytest.mark.parametrize('rank_log',[False,True])
def test_cached_conditioner_accepts_changed_runtime_spectrum_on_all_faces(rank_log):
    if len(jax.devices())!=4:
        pytest.skip('explicit one-process four-CPU-device source fixture')
    mesh=Mesh(np.asarray(jax.devices()).reshape(2,2),('x','y'))
    face=NamedSharding(mesh,P(None,'x','y'));rep=NamedSharding(mesh,P())
    rng=np.random.default_rng(623)
    V,_=np.linalg.qr(rng.normal(size=(8,8))+1j*rng.normal(size=(8,8)))
    Z=(rng.normal(size=(1,8,4))+1j*rng.normal(size=(1,8,4))).astype(np.complex128)
    vectors=jax.device_put(V[None].astype(np.complex128),face)
    z=jax.device_put(Z,face)
    program=cplus._face_conditioner(mesh,1e-8,rank_log,8)
    assert program is cplus._face_conditioner(mesh,1e-8,rank_log,8)
    results=[]
    for eigenvalues,first_kept in (([0,1e-12,1,2,3,4,5,6],2),([0,1e-12,1e-12,2,3,4,5,6],3)):
        lam=np.asarray(eigenvalues,np.float64)
        B=program(jax.device_put(lam[None],rep),vectors)
        value=cplus.apply(B,z,mesh_xy=mesh,panel_bytes=256)
        # Independent literal spectral sum with declared retained columns.
        inverse=np.zeros((8,8),np.complex128)
        for i in range(first_kept,8):inverse+=np.outer(V[:,i],V[:,i].conj())/lam[i]
        np.testing.assert_allclose(np.asarray(value)[0],inverse@Z[0],atol=3e-13,rtol=3e-13)
        np.testing.assert_array_equal(np.asarray(B)[0,:,:first_kept],np.zeros((8,first_kept),np.complex128))
        assert B.sharding==face and value.sharding==face
        assert len(B.addressable_shards)==4
        assert all(s.data.shape==(1,4,4) for s in B.addressable_shards)
        results.append(np.asarray(value))
    assert np.max(abs(results[0]-results[1]))>1e-2

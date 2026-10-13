"""Lattice images retain their physical Bloch phase in the production sampler."""
import numpy as np


def exercise_fractional_lattice_images(mesh):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from common.psi_G_store import _gslot_face_kernel, _gslot_faces_kernel
    from common.wfn_layout import PSI_NMU_ACC_SPEC, PSI_MUNT_ACC_SPEC
    grid=(8,10,12); nk=2; nb=8; ng=8; nm=8
    G=np.array([[0,0,0],[-1,2,-3],[2,-3,1],[-3,1,4],
                [3,-4,-5],[-2,-1,3],[71,82,93],[91,81,71]],np.int32)
    geometry=np.broadcast_to(np.c_[G,[1,1,1,1,1,1,0,0]],(nk,ng,4)).copy()
    k=np.array([[0,0,0],[1/3,-1/3,1/3]])
    base=np.array([[.123,.317,.731],[.019,.815,.415],[-1e-18,.2,.3],[.5,.4,.3]])
    L=np.array([[1,-2,3],[-3,2,1],[1,0,-1],[2,2,2]])
    r=np.concatenate([base,base+L])
    active=np.array([1,1,1,0,1,1,1,0.])
    errors={}
    for ns in (2,4):
        rng=np.random.default_rng(910+ns)
        coeff=rng.normal(size=(nk,nb,ns,ng))+1j*rng.normal(size=(nk,nb,ns,ng))
        coeff[...,6:]=19.+23j
        put=lambda a,s: device_put_process_local(np.asarray(a),NamedSharding(mesh,s))
        fn=_gslot_face_kernel(mesh,grid,nk,nb,ns,ng,nm,2,True,'fractional')
        specs=[((nk,nb,ns,ng),np.complex128,P(None,('x','y'),None,None)),
               (geometry.shape,np.int32,P()),(k.shape,np.float64,P()),
               (r.shape,np.float64,P()),(active.shape,np.float64,P()),
               ((nk,nb,ns,nm),np.complex128,PSI_NMU_ACC_SPEC),
               ((nk,nb,ns,nm),np.complex128,PSI_MUNT_ACC_SPEC),
               ((),np.int32,P())]
        with mesh:
            abstract=[jax.ShapeDtypeStruct(shape,dtype,sharding=NamedSharding(mesh,s))
                      for shape,dtype,s in specs]
            compiled=fn.lower(*abstract).compile()
            z=np.zeros((nk,nb,ns,nm),np.complex128)
            _,y,x=compiled(put(coeff,specs[0][2]),put(geometry,P()),put(k,P()),
                         put(r,P()),put(active,P()),put(z,PSI_NMU_ACC_SPEC),
                         put(z,PSI_MUNT_ACC_SPEC),put(np.int32(0),P()))
            y,x=_gslot_faces_kernel(mesh)(y,x)
            got=np.asarray(gather_to_host(y)); companion=np.asarray(gather_to_host(x))
        phase=np.exp(2j*np.pi*np.einsum('kgi,mi->kgm',k[:,None]+G[None,:6],r))
        truth=np.einsum('knsg,kgm->knsm',coeff[...,:6],phase)
        truth*=active[None,None,None,:]/np.sqrt(np.prod(grid))
        np.testing.assert_allclose(got,truth,atol=3e-14,rtol=3e-13)
        np.testing.assert_array_equal(companion,got.conj().transpose(0,3,1,2))
        translation=np.exp(2j*np.pi*np.einsum('ki,mi->km',k,L))
        np.testing.assert_allclose(got[...,4:],got[...,:4]*translation[:,None,None,:],
                                   atol=3e-14,rtol=3e-13)
        np.testing.assert_allclose(companion[:,4:],companion[:,:4]*translation.conj()[:,:,None,None],
                                   atol=3e-14,rtol=3e-13)
        assert not np.any(got[...,3]) and not np.any(got[...,7])
        assert np.max(abs(got[1,...,4:7]-got[1,...,:3]))>1e-3
        errors[str(ns)]=float(np.max(abs(got-truth)))
    return dict(status='PASS_FRACTIONAL_LATTICE_IMAGE_PRODUCTION_SAMPLER',
                max_absolute_error_by_components=errors,
                mesh_shape={str(k):int(v) for k,v in mesh.shape.items()},
                tested='actual production signed-G kernel; original r Bloch phase, translated Y/X, negative G, tiny negative boundary, populated invalid G, inactive ghosts',
                scope='Exact sampler algebra only; no material FIT or energy certificate')


def test_fractional_lattice_images():
    import jax
    from jax.sharding import Mesh
    mesh=Mesh(np.asarray(jax.devices()[:1]).reshape(1,1),('x','y'))
    exercise_fractional_lattice_images(mesh)

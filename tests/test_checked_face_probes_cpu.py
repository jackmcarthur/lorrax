"""Checked products preserve projected equations without implicit matrix transport."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('XLA_FLAGS','--xla_force_host_platform_device_count=144')
import math,re
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from distrib_la import _result_check as R


def mesh_for(side):
    if len(jax.devices())<side*side:pytest.skip('requires virtual square mesh')
    return Mesh(np.array(jax.devices()[:side*side]).reshape(side,side),('x','y'))


def test_skinny_product_normal_adjoint_broadcast_and_refusals():
    jax.config.update('jax_enable_x64',True);rng=np.random.default_rng(81)
    for side in (2,3,12):
        mesh=mesh_for(side);n,m=side*5,side*7
        for dtype in (np.float64,np.complex128):
            a=rng.normal(size=(2,n,m)).astype(dtype)
            if dtype==np.complex128:a+=1j*rng.normal(size=a.shape)
            for adjoint in (False,True):
                for batched in (False,True):
                    shape=((2,)if batched else())+((n if adjoint else m),8)
                    p=rng.normal(size=shape).astype(dtype)
                    if dtype==np.complex128:p+=1j*rng.normal(size=shape)
                    fn=jax.jit(lambda a,p:R._face_probe_product(a,p,mesh=mesh,adjoint=adjoint))
                    got=fn(a,p);want=(np.swapaxes(a.conj(),-1,-2)if adjoint else a)@p
                    np.testing.assert_allclose(np.asarray(got),want,rtol=2e-14,atol=2e-14)
                    assert got.sharding.is_equivalent_to(NamedSharding(mesh,P(None,'y'if adjoint else'x',None)),3)
        with pytest.raises(ValueError,match='contracted'):
            R._face_probe_product(jnp.ones((n,m)),jnp.ones((m+1,8)),mesh=mesh)
        with pytest.raises(ValueError):
            R._face_probe_product(jnp.ones((2,n,m)),jnp.ones((3,m,8)),mesh=mesh)


def test_eigh_cut_and_solve_checks_keep_independent_equations():
    jax.config.update('jax_enable_x64',True);rng=np.random.default_rng(92)
    for side in (2,3,12):
        mesh=mesh_for(side);n=side*5;nrhs=side*3
        for dtype in (np.float64,np.complex128):
            q=rng.normal(size=(n,n)).astype(dtype)
            if dtype==np.complex128:q+=1j*rng.normal(size=q.shape)
            q=np.linalg.qr(q)[0];w=np.linspace(-.3,1.5,n);a=(q*w)@q.conj().T
            bad=q.copy();bad[:,0]*=1.3
            for cut in (None,.2):
                fn=jax.jit(lambda a,w,q:R.eigh_errors(a,w,q,mesh=mesh,cut=cut))
                good=np.asarray(fn(a,w,q));assert np.max(good)<R.accept(n,dtype)
                damaged=np.asarray(fn(a,w,bad))
                assert (np.max(damaged)>R.accept(n,dtype))if cut is None else(np.max(damaged)<R.accept(n,dtype))
            aa=np.eye(n)+.01*rng.normal(size=(2,n,n)).astype(dtype)
            xx=rng.normal(size=(2,n,nrhs)).astype(dtype)
            if dtype==np.complex128:aa+=.01j*rng.normal(size=aa.shape);xx+=1j*rng.normal(size=xx.shape)
            bb=aa@xx
            fn=jax.jit(lambda a,b,x:R.solve_errors((*R.matrix_sketch(a,mesh),*R.rhs_sketch(b,mesh)),x,mesh))
            np.testing.assert_array_less(np.asarray(fn(aa,bb,xx)),R.accept(n,dtype))
            broken=xx.copy();broken[:,0,0]+=.2
            got=float(np.asarray(fn(aa,bb,broken))[0])
            probes=np.asarray(R.probes(n,dtype,salt=1));ahw=np.swapaxes(aa.conj(),-1,-2)@probes
            whb=probes.conj().T@bb;projected=np.swapaxes(ahw.conj(),-1,-2)@broken-whb
            fro=lambda v:np.sqrt(np.sum(np.abs(v)**2,axis=(-2,-1)))
            expected=np.max(fro(projected)/(np.sqrt(8)*(fro(aa)*fro(broken)+fro(bb))))
            np.testing.assert_allclose(got,expected,rtol=2e-13,atol=2e-15)
            assert got>R.accept(n,dtype)


def test_large_checker_hlo_keeps_matrix_faces_and_owned_reductions():
    jax.config.update('jax_enable_x64',True)
    for side in (2,3,12):
        mesh=mesh_for(side);n=49164 if side==12 else side*257;nrhs=side*13
        for dtype in (jnp.float64,jnp.complex128):
            face=NamedSharding(mesh,P(None,'x','y'));rep=NamedSharding(mesh,P())
            aa=jax.ShapeDtypeStruct((2,n,n),dtype,sharding=face);xx=jax.ShapeDtypeStruct((2,n,nrhs),dtype,sharding=face);w=jax.ShapeDtypeStruct((2,n),jnp.float64,sharding=rep)
            fn=jax.jit(lambda a,w,q,b,x:(R.eigh_errors(a,w,q,mesh=mesh,cut=.1),R.solve_errors((*R.matrix_sketch(a,mesh),*R.rhs_sketch(b,mesh)),x,mesh)))
            text=fn.lower(aa,w,aa,xx,xx).compile().as_text();bound=2*(n//side)**2
            for line in text.splitlines():
                m=re.match(r'\s*(?:ROOT )?%\S+ = (?:f64|c128)\[([0-9,]*)\]',line)
                if m:assert (math.prod(map(int,m[1].split(',')))if m[1]else 1)<=bound,line
                if side>2 and re.search(r' all-reduce(?:-start)?\(',line):
                    assert not re.search(r'(?:f64|c128)\[',line),line

"""Dead-row restoration: literal eigenvector ordering and bounded distributed fields."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=144')
import math
import re
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from distrib_la._result_check import deflate_zero_rows, eigh_layout


def _program(side):
    count=side*side
    if len(jax.devices())<count:pytest.skip(f'needs {count} virtual CPU devices')
    mesh=Mesh(np.array(jax.devices()[:count]).reshape(side,side),('x','y'))
    face=lambda nd:NamedSharding(mesh,P(*((None,)*(nd-2)),'x','y'))
    def provider(a):
        # Exact diagonal eigensolver, generating only the owned eigenvector tile.
        # It receives the real wrapper's sentinel-modified runtime operand.
        diagonal=jnp.real(jnp.diagonal(a,axis1=-2,axis2=-1))
        order=jnp.argsort(diagonal,axis=-1,stable=True)
        values=jnp.take_along_axis(diagonal,order,axis=-1)
        n=a.shape[-1];width=n//side
        @jax.shard_map(mesh=mesh,in_specs=P(),out_specs=face(a.ndim).spec,check_vma=False)
        def columns(indices):
            rows=jax.lax.axis_index('x')*width+jnp.arange(width)
            chosen=jax.lax.dynamic_slice_in_dim(indices,jax.lax.axis_index('y')*width,width,axis=-1)
            phase=(jnp.exp(.13j*chosen) if jnp.issubdtype(a.dtype,jnp.complexfloating)
                   else jnp.where(chosen%2, -1., 1.))
            return (rows[:,None]==chosen[...,None,:])*phase[...,None,:]
        return values,columns(order).astype(a.dtype)
    def build(nd):
        sh=face(nd)
        return jax.jit(deflate_zero_rows(provider,constrain=eigh_layout(mesh,nd),mesh=mesh),
                       in_shardings=sh,out_shardings=(NamedSharding(mesh,P()),sh))
    return mesh,build


def _reference(diagonal,dtype=complex):
    n=diagonal.shape[-1]
    outw=[];outq=[]
    for d in diagonal.reshape(-1,n):
        bound=max(float(np.max(np.abs(d))),1e-300)
        if not np.any(d):bound=1.
        transformed=np.where(d==0,-2*bound*(1+np.arange(n)/n),d)
        first=np.argsort(transformed,kind='stable')
        restored=np.where(transformed[first]<-1.5*bound,0,transformed[first])
        order=np.argsort(restored,kind='stable')
        source=first[order]
        phase=np.exp(.13j*source) if np.issubdtype(np.dtype(dtype),np.complexfloating) else np.where(source%2,-1.,1.)
        outw.append(restored[order]);outq.append(np.eye(n,dtype=dtype)[:,source]*phase[None,:])
    return np.array(outw).reshape(diagonal.shape),np.array(outq).reshape(diagonal.shape+(n,))


def test_negative_and_roundoff_spectrum_with_dead_rows_and_repeated_values():
    jax.config.update('jax_enable_x64',True)
    for side in (2,3,12):
        mesh,build=_program(side);n=side*5  # Prime local tile; crosses arbitrary owners.
        d=np.linspace(.2,1.4,n);d[1]=-.25;d[3]=-1e-18;d[-2:]=0;d[0]=d[2]
        for dtype in (np.float64,np.complex128):
            for diagonal in (d,np.stack((d,np.roll(d,3),np.zeros_like(d)))):
                a=diagonal[...,None,:]*np.eye(n)
                call=build(a.ndim);w,q=map(np.asarray,call(a.astype(dtype)))
                rw,rq=_reference(diagonal,dtype)
                np.testing.assert_allclose(w,rw,rtol=0,atol=0)
                np.testing.assert_allclose(q,rq,rtol=0,atol=2e-16)
                np.testing.assert_allclose(a@q,q*w[...,None,:],rtol=0,atol=2e-16)
                np.testing.assert_allclose(np.swapaxes(q.conj(),-1,-2)@q,np.broadcast_to(np.eye(n),q.shape),rtol=0,atol=3e-16)


def test_reorder_hlo_keeps_both_column_directions_in_local_faces():
    jax.config.update('jax_enable_x64',True)
    for side in (2,3,12):
        _,build=_program(side);n=49164 if side==12 else side*257
        # Actual wrapper and dynamic diagonal provider, no matrix allocation.
        for batch in ((),(2,)):
            shape=batch+(n,n);exe=build(len(shape)).lower(jax.ShapeDtypeStruct(shape,jnp.complex128)).compile()
            text=exe.as_text();local=math.prod(batch or (1,))*(n//side)**2
            linear=math.prod(batch or (1,))*n
            sizes=[]
            for line in text.splitlines():
                m=re.match(r'\s*(?:ROOT )?%\S+ = c128\[([0-9,]*)\]',line)
                if m:sizes.append(math.prod(map(int,m[1].split(',')))if m[1]else 1)
            assert sizes and max(sizes)<=max(local,linear),(side,batch,max(sizes),local,linear)
            assert 'collective-permute' in text

"""Spin-GGA analytic and independent directional-energy derivative controls."""
import io
import os
import subprocess
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from common.fft_helpers import local_fftn3
from psp.xc import (compute_V_xc_noncollinear,pbe_functional_polarized,
    pbe_functional_polarized_components)


def geometry(n=8):
    k=np.arange(n);k=np.where(k<n//2,k,k-n)
    g=np.stack(np.meshgrid(k,k,k,indexing="ij"),axis=-1)
    xyz=np.stack(np.meshgrid(*(np.arange(n)*2*np.pi/n for _ in range(3)),indexing="ij"),axis=0)
    # Explicit tiny spectral derivative matrix, independent of the FFT owner.
    i,j=np.meshgrid(np.arange(n),np.arange(n),indexing="ij")
    d=np.real(np.sum(1j*k[:,None,None]*np.exp(2j*np.pi*k[:,None,None]*(i-j)/n),axis=0)/n)
    def gradients(field):
        return [np.moveaxis(np.tensordot(d,field,axes=(1,a)),0,a) for a in range(3)]
    return g,xyz,gradients


def fields():
    g,r,gradient=geometry();x,y,z=r
    rho=1.+.08*np.cos(x)+.05*np.sin(y)
    amp=.2+.04*np.sin(x)+.03*np.cos(y)
    direction=np.asarray([1.,2.,3.])/np.sqrt(14.)
    mag=direction[:,None,None,None]*amp
    return g,r,gradient,rho,mag


def inputs(rho,mag,gradient):
    m=np.sqrt(np.sum(mag*mag,axis=0))
    u,d=(rho+m)/2,(rho-m)/2
    gu,gd=gradient(u),gradient(d)
    return u,d,sum(a*a for a in gu),sum(a*b for a,b in zip(gu,gd)),sum(b*b for b in gd)


def potential(rho,mag,g,fn,components=None):
    v,b=compute_V_xc_noncollinear(jnp.asarray(rho),local_fftn3(jnp.asarray(rho)),
        jnp.asarray(mag),jnp.asarray(g,dtype=jnp.float64),fn,100.,xc_components=components)
    return np.asarray(v),np.asarray(b)


@pytest.mark.parametrize("sign",[1.,-1.])
def test_constant_charge_spin_gradient_exact_field(sign):
    g,r,gradient=geometry();rho=np.ones(g.shape[:-1]);mag=np.zeros((3,)+rho.shape)
    mag[2]=sign*(.2+.05*np.cos(r[0]))
    def energy_per_particle(u,d,suu,sud,sdd):return (suu+sdd)/(u+d)
    v,b=potential(rho,mag,g,energy_per_particle)
    expected=np.zeros_like(b);expected[2]=sign*.05*np.cos(r[0])
    np.testing.assert_allclose(v,0.,atol=2e-13,rtol=0)
    np.testing.assert_allclose(b,expected,atol=2e-13,rtol=0)


def test_spin_gradient_global_rotation_covariance():
    g,r,gradient,rho,mag=fields()
    def fn(u,d,suu,sud,sdd):return (u**3+2*d**3+(.3+u)*suu+.2*sud+(.5+d)*sdd)/(u+d)
    rotation=np.asarray([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    v,b=potential(rho,mag,g,fn);vm,bm=potential(rho,np.einsum("ab,bxyz->axyz",rotation,mag),g,fn)
    np.testing.assert_allclose(vm,v,atol=2e-12,rtol=2e-12)
    np.testing.assert_allclose(bm,np.einsum("ab,bxyz->axyz",rotation,b),atol=2e-12,rtol=2e-12)


@pytest.mark.parametrize("direction",["charge","radial_spin","transverse_spin","coupled"])
def test_coupled_gradient_density_potential_is_energy_derivative(direction):
    g,r,gradient,rho,mag=fields();x,y,z=r
    def fn(u,d,suu,sud,sdd):return (u**3+2*d**3+(.3+u)*suu+.2*sud+(.5+d)*sdd)/(u+d)
    drho=.3*np.cos(x)+.2*np.sin(y)
    dm=np.asarray([.2*np.sin(y),.3*np.cos(x),.1*np.sin(x+y)])
    if direction=="charge":dm*=0
    if direction=="radial_spin":drho*=0;dm=mag*.3*np.cos(x)[None]
    if direction=="transverse_spin":
        drho*=0;dm=np.cross(np.moveaxis(mag,0,-1),np.asarray([1.,-.5,.3]));dm=np.moveaxis(dm,-1,0)
    v,b=potential(rho,mag,g,fn)
    predicted=float(np.sum(v*drho)+np.sum(b*dm))
    def energy(r,m):
        u,d,suu,sud,sdd=inputs(r,m,gradient)
        return float(np.sum((u+d)*fn(u,d,suu,sud,sdd)))
    h=1e-5;coarse=(energy(rho+h*drho,mag+h*dm)-energy(rho-h*drho,mag-h*dm))/(2*h)
    h/=2;fine=(energy(rho+h*drho,mag+h*dm)-energy(rho-h*drho,mag-h*dm))/(2*h)
    assert abs(fine-predicted)<2e-7+2e-8*abs(predicted)
    assert abs(fine-coarse)<3e-7+2e-8*abs(predicted)


def libxc(component,points):
    executable=os.environ.get("CDREF_LIBXC_ORACLE")
    if not executable:pytest.skip("separately authenticated Libxc7 oracle fixture not selected")
    points=np.asarray(points,dtype=float).reshape(-1,5)
    payload=f"{len(points)}\n"+"\n".join(" ".join(f"{v:.17g}" for v in row) for row in points)+"\n"
    output=subprocess.check_output([executable,component],input=payload,text=True)
    rows=np.loadtxt(io.StringIO(output),ndmin=2)
    if rows.shape!=(len(points),6) or not np.isfinite(rows).all():raise ValueError("Libxc oracle shape/finite refusal")
    return rows*2. # independent provider Hartree → owner Rydberg


@pytest.mark.parametrize("component",["exchange","correlation"])
def test_pbe_component_values_and_all_derivatives_against_libxc(component):
    pytest.importorskip("jax_xc.impl")
    fx,fc=pbe_functional_polarized_components();fn=fx if component=="exchange" else fc
    points=np.asarray([[.6,.4,.02,.005,.01],[.7,.3,.012,-.002,.015],[.4,.35,.007,.003,.01]])
    expected=libxc(component,points)
    def energy(p):return (p[0]+p[1])*fn(*p)
    actual_e=np.asarray(jax.vmap(lambda p:fn(*p))(jnp.asarray(points)))
    derivative=np.asarray(jax.vmap(jax.grad(energy))(jnp.asarray(points)))
    np.testing.assert_allclose(actual_e,expected[:,0],atol=2e-11,rtol=2e-10)
    np.testing.assert_allclose(derivative,expected[:,1:],atol=2e-10,rtol=3e-9)


@pytest.mark.parametrize("component",["exchange","correlation","combined"])
def test_actual_pbe_potential_directional_libxc_energy_derivative(component):
    pytest.importorskip("jax_xc.impl")
    fx,fc=pbe_functional_polarized_components()
    g,r,gradient,rho,mag=fields();x,y,z=r
    def zero(u,d,suu,sud,sdd):return jnp.zeros_like(u)
    xp,cp=(fx if component!="correlation" else zero),(fc if component!="exchange" else zero)
    fn=lambda *a:xp(*a)+cp(*a)
    v,b=potential(rho,mag,g,fn,(xp,cp))
    drho=.3*np.cos(x)+.2*np.sin(y);dm=np.asarray([.2*np.sin(y),.3*np.cos(x),.1*np.sin(x+y)])
    predicted=float(np.sum(v*drho)+np.sum(b*dm))
    def energy(r,m):
        points=np.stack(inputs(r,m,gradient),axis=-1);total=0.
        for name in ("exchange","correlation"):
            if component in (name,"combined"):
                total+=float(np.sum((points[...,0]+points[...,1]).ravel()*libxc(name,points)[:,0]))
        return total
    h=1e-5;coarse=(energy(rho+h*drho,mag+h*dm)-energy(rho-h*drho,mag-h*dm))/(2*h)
    h/=2;fine=(energy(rho+h*drho,mag+h*dm)-energy(rho-h*drho,mag-h*dm))/(2*h)
    assert abs(predicted-fine)<2e-7+3e-8*abs(predicted)
    assert abs(coarse-fine)<3e-7+3e-8*abs(predicted)


def test_public_pbe_combined_api_preserved():
    pytest.importorskip("jax_xc.impl")
    fx,fc=pbe_functional_polarized_components();fn=pbe_functional_polarized()
    p=(jnp.asarray([.6,.7]),jnp.asarray([.4,.3]),.02,.005,.01)
    np.testing.assert_array_equal(fn(*p),fx(*p)+fc(*p))

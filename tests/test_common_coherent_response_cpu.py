"""Tiny coherent endpoints through the incumbent direct/mode-11 owner.

CPU reference-backend algebra coverage, not native GPU qualification.
"""
from types import SimpleNamespace

import numpy as np
import pytest


def _fixture(monkeypatch):
    import jax
    from jax.sharding import Mesh
    from tests import test_tdagw_dynsig_sigma as fixture
    jax.config.update("jax_enable_x64", True)
    dev = np.asarray(jax.devices("cpu"))
    mesh = Mesh(dev[:4].reshape(2,2) if len(dev)>=4 else dev[:1].reshape(1,1), ("x","y"))
    grid=(4,2,1); nk,nb,ns,mu=8,4,2,4
    for name,value in dict(KGRID=grid,NBF=nb,NB_PAD=nb,NS=ns,NMU=mu,NOCC=2).items():
        monkeypatch.setattr(fixture,name,value)
    rng=np.random.default_rng(14421)
    psi=.15*(rng.normal(size=(nk,nb,ns,mu))+1j*rng.normal(size=(nk,nb,ns,mu)))
    energy=np.array([-1.2,-.3,.4,1.1])[None]+rng.uniform(-.04,.04,(nk,nb))
    rho=np.zeros((nk,nb,nb),complex);rho[:,0,0]=1
    for k in range(nk):
        u,_=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))
        rho[k,1:3,1:3]=u@np.diag([.23,.78])@u.conj().T
    kf=np.indices(grid).reshape(3,-1).T/np.array(grid)
    plan,perm=fixture._plan(mesh,kf,rng)
    psi=psi[...,perm]
    wf,_=fixture._gw_bundle(mesh,psi,energy,rho.diagonal(axis1=1,axis2=2).real,plan)
    return mesh,grid,psi,energy,rho,wf.green_parent


@pytest.mark.parametrize("qgrid",[(4,2,1),(2,2,1)])
def test_common_native_parent_endpoint_stream_all_blocks_and_current_inputs(monkeypatch,qgrid):
    import jax.numpy as jnp
    from ffi.fft import xla_reference
    from gw.greens_function_kernel import density_left_endpoint
    from gw.w_isdf import full_density_response_stream
    from tests._full_density_blocks_oracle import four_block_panels,pair_spectrum,pair_time
    from tests.test_chi_real_space_crop_cpu import _literal_crop
    mesh,grid,psi,energy,rho,parent=_fixture(monkeypatch)
    nk,nb,ns,mu=psi.shape
    times=np.array([0.,.13+.09j,-.11+.17j,.05-.08j])
    weights=np.array([[[.2+.1j,.1-.2j,-.3+.2j,.11]],
                      [[-.1+.2j,.3+.1j,.2-.1j,-.07j]]])
    origin=.173
    coordinates=np.indices(grid).reshape(3,-1).T
    partners=np.array([np.ravel_multi_index(((coordinates-q)%grid).T,grid) for q in coordinates])
    endpoints=density_left_endpoint(mesh,parent.psi_mun.shape)
    for changed in (False,True):
        current=rho.copy()
        if changed:
            current[:,1:3,1:3]=np.eye(2)-current[:,1:3,1:3]
        blocks=four_block_panels(current,slice(1,3))
        for name,(left,right) in blocks.items():
            l=endpoints(parent.psi_mun,jnp.asarray(left))
            r=endpoints(parent.psi_mun,jnp.asarray(right))
            supports=[np.any(matrix!=0,axis=1) for matrix in (left,right)]
            bounds=tuple(value for mask in supports
                         for value in (energy[mask].min(),energy[mask].max()))
            bands=np.arange(nb)[None]
            ranges=np.array([[np.min(np.where(mask,bands,nb),axis=1),
                              np.max(np.where(mask,bands+1,0),axis=1)] for mask in supports])
            common=dict(q_ids=tuple(range(np.prod(qgrid))),n_outputs=1,
                        pair_bounds_ry=bounds,pair_band_ranges=ranges,
                        rule_reference_ry=origin,qgrid=qgrid)
            with xla_reference():
                native=full_density_response_stream(mesh,grid,(nk,nb,mu,ns),
                    k_unfold_plan=parent.plan,**common)
                fallback=full_density_response_stream(mesh,grid,(nk,nb,mu,ns),**common)
                args=(jnp.asarray(times),jnp.asarray(weights),l,parent.psi_nmu,r,parent.enk)
                got=np.asarray(native(*args))[:,0]
                unfolded=np.asarray(fallback(*args))[:,0]
            np.testing.assert_allclose(got,unfolded,rtol=5e-12,atol=2e-14)
            expected=[]
            for partner in partners:
                f=pair_time(pair_spectrum(psi,energy,left,right,partner),times)
                b=pair_time(pair_spectrum(psi,energy,right,left,partner),-times.conjugate())
                expected.append(np.einsum('t,tij->ij',weights[0,0]*np.exp(origin*times),f)
                                +np.einsum('t,tij->ij',weights[1,0]*np.exp(origin*times.conjugate()),b))
            expected=np.asarray(expected)*np.sqrt(nk)
            if qgrid!=grid:
                er=np.fft.ifftn(expected.reshape(*grid,mu,mu),axes=(0,1,2),norm='ortho').reshape(expected.shape)
                er=_literal_crop(er,grid,qgrid)
                expected=np.fft.fftn(er.reshape(*qgrid,mu,mu),axes=(0,1,2),norm='ortho').reshape(-1,mu,mu)*np.sqrt(np.prod(qgrid)/nk)
            np.testing.assert_allclose(got,expected,rtol=5e-12,atol=2e-14,err_msg=name)


def test_native_parent_asymmetric_moments_preserve_independent_insertions(monkeypatch):
    import jax.numpy as jnp
    from ffi.fft import xla_reference
    from gw.greens_function_kernel import density_left_endpoint
    from gw.response_bank import full_density_bare_moments
    mesh,grid,psi,energy,rho,parent=_fixture(monkeypatch)
    ph=density_left_endpoint(mesh,parent.psi_mun.shape)(parent.psi_mun,jnp.asarray(rho))
    meta=SimpleNamespace(nkx=4,nky=2,nkz=1,qgrid=grid)
    calls=[]
    def execute(program,args,stage,runtime_bytes):
        if not calls:
            import jax
            from gw.w_isdf import _BoundTail
            assert isinstance(program,_BoundTail) and program.tail
            # The OUTER adapter receives the native tables as operands too;
            # no closed-over distributed jax.Array may cross this jit.
            closed=jax.make_jaxpr(program.program)(*args,*program.tail)
            tables=jax.tree.leaves(program.tail)
            assert all(const is not table for const in closed.consts for table in tables)
            assert not {'common','common_tail'} & set(program.program.__wrapped__.__code__.co_freevars)
        calls.append((stage,runtime_bytes,len(args)))
        return program(*args)
    with xla_reference():
        args=((parent.psi_mun,parent.psi_nmu,ph,parent.enk),meta)
        expected=full_density_bare_moments(*args,mesh_xy=mesh,q_ids=tuple(range(8)))
        got=full_density_bare_moments(*args,mesh_xy=mesh,q_ids=tuple(range(8)),
                                     k_unfold_plan=parent.plan,execute=execute)
    for a,b in zip(got,expected):
        np.testing.assert_allclose(a,b,rtol=4e-12,atol=2e-14)
    assert [c[0] for c in calls]==['direct','moment_correlation','moment_correlation']
    assert all(c[1]==0 and c[2]==9 for c in calls)


def test_endpoint_mode_refuses_ambiguous_prepared_ranges():
    from gw.w_isdf import _get_chi_fractional_contour_kernel_face
    from tests.test_chi_real_space_crop_cpu import _mesh
    with pytest.raises(ValueError,match='runtime endpoint band_ranges'):
        _get_chi_fractional_contour_kernel_face(_mesh(),(1,1,1),1,(1,2,4,2),
            selected_q=(0,),pair_mode='direct',ordered=True,endpoint_pair=True,
            band_ranges=((0,1),(1,2)))


def test_equilibrium_default_graph_keeps_exact_output_bits(monkeypatch):
    from ffi.fft import xla_reference
    from gw.w_isdf import _get_chi_fractional_contour_kernel_face
    mesh,grid,_,_,_,parent=_fixture(monkeypatch)
    options=dict(k_unfold_plan=parent.plan,selected_q=tuple(range(8)),
                 pair_mode='direct',ordered=True)
    times=np.array([.1+.05j,-.07+.11j])
    weights=np.array([[[.3+.2j,.1]],[[.2-.1j,-.07j]]])
    args=(times,weights,parent.psi_mun,parent.psi_nmu,parent.enk,
          parent.occ,1-parent.occ,np.array([0.,0.]))
    with xla_reference():
        legacy=_get_chi_fractional_contour_kernel_face(mesh,grid,1,(8,4,4,2),**options)
        explicit=_get_chi_fractional_contour_kernel_face(mesh,grid,1,(8,4,4,2),
                                                       endpoint_pair=False,**options)
        np.testing.assert_array_equal(legacy(*args),explicit(*args))


def compare_parent_equilibrium_bits(parent_revision='21fcc77'):
    """Opt-in CPU evidence: execute the actual git-parent owner in isolation.

    Not a collected test: needs the preserved git object. No checkout, file
    mutation, native rebuild or monkeypatch of the contraction owner.
    """
    import subprocess
    import sys
    import types
    from runtime.source_closure import ensure_source_closure
    ensure_source_closure()
    from ffi.fft import xla_reference
    from gw import w_isdf
    source=subprocess.run(['git','show',f'{parent_revision}:src/gw/w_isdf.py'],
                          check=True,capture_output=True,text=True).stdout
    old=types.ModuleType('gw._equilibrium_parent_evidence');old.__package__='gw'
    sys.modules[old.__name__]=old
    exec(compile(source,f'git:{parent_revision}:src/gw/w_isdf.py','exec'),old.__dict__)
    with pytest.MonkeyPatch.context() as patch:
        mesh,grid,_,_,_,parent=_fixture(patch)
        options=dict(k_unfold_plan=parent.plan,selected_q=tuple(range(8)),pair_mode='direct',ordered=True)
        args=(np.array([.1+.05j,-.07+.11j]),np.array([[[.3+.2j,.1]],[[.2-.1j,-.07j]]]),
              parent.psi_mun,parent.psi_nmu,parent.enk,parent.occ,1-parent.occ,np.array([0.,0.]))
        with xla_reference():
            a=old._get_chi_fractional_contour_kernel_face(mesh,grid,1,(8,4,4,2),**options)
            b=w_isdf._get_chi_fractional_contour_kernel_face(mesh,grid,1,(8,4,4,2),**options)
            left,right=np.asarray(a(*args)),np.asarray(b(*args))
        np.testing.assert_array_equal(left.view(np.uint64),right.view(np.uint64))
    return dict(parent=parent_revision,shape=list(left.shape),uint64_equal=True,
                scope='CPU typed-parent scalar direct fixture; not native CUDA/material')

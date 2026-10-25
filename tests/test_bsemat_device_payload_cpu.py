"""The export's device gauge/rotation equals its literal coefficient equations."""
import runtime
from runtime.source_closure import ensure_source_closure
ensure_source_closure()
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import os
import ast
import re
from pathlib import Path
from functools import partial
from unittest.mock import patch
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from bse.bsemat_export import _canonical_psi, _gauge_unitaries, _rotate_psi, _identity_gauge

jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize('side,ns', [(2,1), (2,2), (3,2)])
def test_device_gauge_and_rotation_match_literal_with_poisoned_padding(side, ns):
    if jax.device_count() < side*side:
        pytest.skip('requires nine virtual CPU devices')
    import distrib_la as dla
    mesh = Mesh(np.array(jax.devices()[:side*side]).reshape(side,side), ('x','y'))
    sh = NamedSharding(mesh, P(None,None,None,('x','y')))
    nk, nb, nbg, w0, logical = 2, 4, 6, 1, 13
    mu, band = -(-logical//(side*side))*(side*side), -(-nbg//(side*side))*(side*side)
    rng = np.random.default_rng(912+side+ns)
    af, us = [], []
    for k in range(nk):
        a = rng.normal(size=(logical*ns,nbg))+1j*rng.normal(size=(logical*ns,nbg))
        mixing=np.diag(np.linspace(.4,2.,nbg))+np.triu(
            .12*(rng.normal(size=(nbg,nbg))+1j*rng.normal(size=(nbg,nbg))),1)
        af.append(np.linalg.qr(a)[0]@mixing)
        blocks = [np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))[0]
                  for _ in range(2)]
        u = np.zeros((nb,nb),complex)
        u[:2,:2], u[2:,2:] = blocks
        us.append(u)
    af, us = np.stack(af), np.stack(us)
    a = af.reshape(nk,logical,ns,nbg).transpose(0,3,2,1)
    b = np.einsum('krn,knm->krm',af[:,:,w0:w0+nb],us).reshape(nk,logical,ns,nb).transpose(0,3,2,1)
    ap = np.pad(a,((0,0),(0,band-nbg),(0,0),(0,mu-logical)),constant_values=np.nan)
    bp = np.pad(b,((0,0),(0,0),(0,0),(0,mu-logical)),constant_values=np.nan)
    ad = _canonical_psi(jax.device_put(ap,sh),mesh,n_mu_logical=logical,n_mu_out=mu)
    bd = _canonical_psi(jax.device_put(bp,sh),mesh,n_mu_logical=logical,n_mu_out=mu)
    original_plan = dla.plan
    def unit_plan(op, mesh, **kw):
        kw['backend'] = 'off'
        return original_plan(op,mesh,**kw)
    original_np = np.asarray
    def no_payload_host(a, *args, **kw):
        if isinstance(a,jax.Array) and a.ndim>=3:
            raise AssertionError('export attempted a global tensor host conversion')
        return original_np(a,*args,**kw)
    with patch.object(dla,'plan',side_effect=unit_plan), patch('lxkit.memory.room',return_value=1<<40), \
         patch.object(np,'asarray',side_effect=no_payload_host):
        u, figures = _gauge_unitaries(ad,bd,nbg=nbg,w0=w0,nv=2,mesh_xy=mesh)
        got = _rotate_psi(bd,u,mesh)
    np.testing.assert_allclose(np.asarray(u)[:,:nb,:nb],us,rtol=2e-12,atol=2e-12)
    assert max(figures.values()) < 2e-12
    wanted = np.einsum('knsr,kmn->kmsr',b,np.conj(us))
    expected = np.pad(wanted,((0,0),(0,0),(0,0),(0,mu-logical)))
    np.testing.assert_allclose(np.asarray(got),expected,rtol=3e-12,atol=3e-12)
    assert not got.sharding.is_fully_replicated
    identity = _identity_gauge(nk,nb,mesh)
    np.testing.assert_allclose(np.asarray(_rotate_psi(bd,identity,mesh)),np.nan_to_num(bp),rtol=2e-13,atol=2e-13)


@pytest.mark.parametrize('side,ns', [(2,1), (2,2), (3,2)])
def test_sharded_export_rows_match_literal_all_four_parts(side,ns):
    if jax.device_count() < side*side:
        pytest.skip('requires nine virtual CPU devices')
    from bse.bsemat_export import _psi_views, _row_programs
    from bse import bsemat_algebra as A
    mesh=Mesh(np.array(jax.devices()[:side*side]).reshape(side,side),('x','y'))
    nk,nv,nc,mu=2,3,2,side*side*5
    rng=np.random.default_rng(403+side+ns)
    draw=lambda shape:rng.normal(size=shape)+1j*rng.normal(size=shape)
    v,c=draw((nk,nv,ns,mu)),draw((nk,nc,ns,mu))
    vp=np.pad(v,((0,0),(0,-(-nv//side)*side-nv),(0,0),(0,0)))
    cp=np.pad(c,((0,0),(0,-(-nc//side)*side-nc),(0,0),(0,0)))
    sh=NamedSharding(mesh,P(None,None,None,('x','y')))
    vx,vy=_psi_views(jax.device_put(vp,sh),mesh)
    cx,cy=_psi_views(jax.device_put(cp,sh),mesh)
    w=draw((mu,mu,nk)); z,y=draw((nk,mu)),draw((nk,mu))
    head=np.array([.7,1.2]); iq=np.array([[0,1],[1,0]],np.int32)
    lam=np.array([[1.,2.],[2.,1.]])
    face=NamedSharding(mesh,P('x','y',None)); rep=NamedSharding(mesh,P())
    zd=NamedSharding(mesh,P(None,('x','y')))
    wd,zdv,ydv=jax.device_put(w,face),jax.device_put(z,zd),jax.device_put(y,zd)
    args=(vx,vy,cx,cy,wd,jax.device_put(iq,rep),jax.device_put(iq,rep),zdv,ydv,
          jax.device_put(head,rep),jax.device_put(lam,rep))
    direct,exchange=_row_programs(mesh)
    cv=A.pair_amplitudes(c,v)
    vb=draw((mu,mu)); tc=np.einsum('kcvm,mn->kcvn',cv,vb)
    cvp=A.pair_amplitudes(cp,vp)
    tcp=np.einsum('kcvm,mn->kcvn',cvp,vb)
    for kp in range(nk):
        cc=A.pair_amplitudes(c,np.broadcast_to(c[kp],c.shape))
        vv=A.pair_amplitudes(v,np.broadcast_to(v[kp],v.shape))
        exact=A.direct_parts(cc,vv,np.moveaxis(w[:,:,iq[:,kp]],2,0),z[iq[:,kp]],y[iq[:,kp]],
                             head[iq[:,kp]],omega=7.,lam_q=lam[:,kp])
        executable=direct.lower(*args,kp,omega=7.).compile()
        observed=executable(*args,kp)
        full='c128['+','.join(map(str,(nk,cp.shape[1],cp.shape[1],vp.shape[1],vp.shape[1])))+']'
        assert full not in executable.as_text()
        # The old unconstrained Ccc@W contraction gathered the full mu
        # height of W while retaining only its column-axis partition.
        # The actual fixed SUMMA program must keep that mu^2 field tiled.
        for shape in ((nk,mu,mu//side),(nk,mu//side,mu),(nk,mu,mu)):
            large='c128['+','.join(map(str,shape))+']'
            assert large not in executable.as_text()
        for parts,want in zip(observed,exact):
            got=parts[0,...,0]+1j*parts[0,...,1]
            want=want.transpose(0,2,1,4,3)
            np.testing.assert_allclose(np.asarray(got)[:,:nc,:nc,:nv,:nv],want,rtol=3e-13,atol=3e-12)
            assert parts.sharding.is_equivalent_to(NamedSharding(mesh,P(None,None,None,'x',None,'y',None)),7)
            assert not parts.sharding.is_fully_replicated
        got=exchange(jax.device_put(tcp,NamedSharding(mesh,P(None,None,None,'y'))),
                     jax.device_put(cvp,NamedSharding(mesh,P(None,None,None,'y'))),kp,omega=7.)
        want=np.einsum('kcvn,dun->kdcuv',tc,np.conj(cv[kp]))*7./(8*np.pi)
        got=got[0,...,0]+1j*got[0,...,1]
        np.testing.assert_allclose(np.asarray(got)[:,:nc,:nc,:nv,:nv],want,rtol=3e-13,atol=3e-12)


@pytest.mark.parametrize('side', [2,3])
def test_shared_downfold_slice_has_zero_tail_and_literal_selection(side):
    if jax.device_count()<side*side:
        pytest.skip('requires nine virtual CPU devices')
    from gw.downfold import slice_psi_to_centroids
    mesh=Mesh(np.array(jax.devices()[:side*side]).reshape(side,side),('x','y'))
    mu,nb,ns=side*side*2,5,2
    raw=np.arange(2*nb*ns*mu).reshape(2,nb,ns,mu).astype(complex)*(1+.2j)
    keep=np.array([mu-1,0,side,mu//2,2],np.int32)
    width=-(-len(keep)//(side*side))*(side*side)
    sx=NamedSharding(mesh,P(None,'x',None,None));sy=NamedSharding(mesh,P(None,None,None,'y'))
    px=jax.device_put(np.conj(raw).transpose(0,3,1,2),sx)
    py=jax.device_put(raw,sy)
    x,y=slice_psi_to_centroids(px,py,keep,width,mesh)
    expected=np.pad(raw[...,keep],((0,0),(0,0),(0,0),(0,width-len(keep))))
    np.testing.assert_array_equal(np.asarray(y),expected)
    np.testing.assert_array_equal(np.asarray(x),np.conj(expected).transpose(0,3,1,2))


def test_array_head_equation_matches_report_wrapper():
    from bse import bsemat_algebra as A
    rng=np.random.default_rng(251)
    u=rng.normal(size=(7,7))+1j*rng.normal(size=(7,7))
    v=u@u.conj().T+np.eye(7);w=.6*v;z=rng.normal(size=7)+1j*rng.normal(size=7)
    a=np.linalg.solve(v,np.conj(z))
    actual=jax.jit(lambda v,w,z,a:A.head_row_values(v,w,z,a,.9,xp=jnp))(v,w,z,a)
    reference=A.head_rows(v,w,z,a,.9,rank=7)
    np.testing.assert_allclose(np.asarray(actual[1]),reference.y,rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(np.array([actual[0],actual[2],actual[3],actual[4]],float),
        [reference.W11,reference.figures['solve_residual'],reference.figures['head_row'],reference.figures['s']],
        rtol=1e-13,atol=1e-13)


def test_actual_downfold_export_stage_keeps_tensor_payloads_on_devices():
    if jax.device_count()<4:
        pytest.skip('requires four virtual CPU devices')
    import distrib_la as dla
    from ffi import fft as F
    from bse.bsemat_export import _downfold_stage
    mesh=Mesh(np.array(jax.devices()[:4]).reshape(2,2),('x','y'))
    mu,logical,nb,nq=16,13,4,3
    rng=np.random.default_rng(876)
    psi=rng.normal(size=(nq,nb,1,mu))+1j*rng.normal(size=(nq,nb,1,mu))
    psi[...,logical:]=0
    sh=NamedSharding(mesh,P(None,None,None,('x','y')))
    psid=jax.device_put(psi,sh)
    eye=np.eye(mu,dtype=complex);eye[logical:,logical:]=0
    v=np.repeat(eye[:,:,None],nq,axis=2);w=.7*v
    mat=NamedSharding(mesh,P('x','y',None)); face=NamedSharding(mesh,P('x','y'))
    vd,wd=jax.device_put(v,mat),jax.device_put(w,mat)
    z=np.zeros((nq,mu),complex);z[:,0]=1
    zd=jax.device_put(z,NamedSharding(mesh,P(None,('x','y'))))
    yd=jax.device_put(np.zeros_like(z),zd.sharding)
    original_plan=dla.plan
    def unit_plan(op,mesh,**kw):
        kw['backend']='off'
        return original_plan(op,mesh,**kw)
    original_np=np.asarray
    def no_payload_host(a,*args,**kw):
        if isinstance(a,jax.Array) and (a.ndim>=3 or (a.ndim==2 and a.dtype.kind=='c')):
            raise AssertionError('export downfold attempted a global tensor host conversion')
        return original_np(a,*args,**kw)
    with F.xla_reference(), patch.object(dla,'plan',side_effect=unit_plan), \
         patch('file_io.restart_bundle.read_interaction_orbits',return_value=None), \
         patch.dict(os.environ,{'LORRAX_BANDS_GEMM_FFI':'0'}), \
         patch.object(np,'asarray',side_effect=no_payload_host):
        result=_downfold_stage(mu_small=6,rcond=1e-9,restart='unused',psi_win=psid,
            psi_1=psid,psi_2=psid,n_rmu=logical,W_flat=wd,V_flat=vd,z_keys=zd,y_keys=yd,
            W11=np.full(nq,.7),keys=[(q,(0,0,0)) for q in range(nq)],v_head_q=np.ones(nq),
            Vbar=jax.device_put(eye,face),grid=np.array([nq,1,1]),mesh_xy=mesh)
        jax.block_until_ready(result)
    assert result['fig']['child_solve_residual_max']<2e-12
    for name in ('W_flat','Vbar','psi_1','psi_2','z_keys','y_keys'):
        assert isinstance(result[name],jax.Array)
        assert not result[name].sharding.is_fully_replicated


def test_sharded_row_write_preserves_bgw_shape_and_clips_only_padding(tmp_path):
    if jax.device_count()<9:
        pytest.skip('requires nine virtual CPU devices')
    import h5py
    from file_io.slab_io import SlabIO
    from file_io import bgw_bsemat as F
    mesh=Mesh(np.array(jax.devices()[:9]).reshape(3,3),('x','y'))
    nk,nc,nv=2,2,4
    shape=F.matrix_shape(nk,nv,nc)
    path=tmp_path/'bsemat.h5'
    with h5py.File(path,'w') as f:
        f.create_group('mats')
        for name in F.MATRICES:
            f.create_dataset(name,shape=shape,dtype=np.float64)
    local=np.arange(1*nk*3*3*6*6*2,dtype=float).reshape(1,nk,3,3,6,6,2)
    sh=NamedSharding(mesh,P(None,None,'x',None,'y',None,None))
    data=jax.device_put(local,sh)
    with SlabIO(path,mode='a',mesh=mesh) as io:
        F.register_matrices(io,shape)
        for row in range(nk):
            for name in F.MATRICES:
                io.write_slab(name,data,offset=(row,0,0,0,0,0,0))
        io.sync_writes()
    with h5py.File(path,'r') as f:
        for name in F.MATRICES:
            assert f[name].shape==shape
            expected=np.repeat(local[:,:,:nc,:nc,:nv,:nv],nk,axis=0)
            np.testing.assert_array_equal(f[name][()],expected)


def test_q0_rank_update_keeps_device_rows_and_exact_logical_mask():
    if jax.device_count()<4:
        pytest.skip('requires four virtual CPU devices')
    from bse.bsemat_export import _q0_update_operands
    mesh=Mesh(np.array(jax.devices()[:4]).reshape(2,2),('x','y'))
    sh=NamedSharding(mesh,P(None,('x','y')))
    a=np.arange(2*16).reshape(2,16).astype(complex)
    a[:,13:]=np.nan
    d=np.eye(2,dtype=complex)
    expected=np.nan_to_num(a)
    for source in (a,jax.device_put(a,sh)):
        rows,coeff=_q0_update_operands((source,d),mesh,n_mu=13,n_mu_pad=16)
        np.testing.assert_array_equal(np.asarray(rows),expected)
        np.testing.assert_array_equal(np.asarray(coeff),d)
        assert rows.sharding.is_equivalent_to(sh,2)


@pytest.mark.parametrize('side', [2,3])
def test_actual_placement_programs_never_materialize_global_wavefunction(side):
    if jax.device_count()<side*side:
        pytest.skip('requires nine virtual CPU devices')
    mesh=Mesh(np.array(jax.devices()[:side*side]).reshape(side,side),('x','y'))
    mu=side*side*2
    raw=(np.arange(5*7*2*mu).reshape(5,7,2,mu)+1j).astype(complex)
    root=ast.parse((Path(__file__).parents[1]/'src/bse/bsemat_export.py').read_text())
    def program(owner,name,namespace):
        outer=next(n for n in root.body if isinstance(n,ast.FunctionDef) and n.name==owner)
        node=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name==name)
        env=dict(jax=jax,jnp=jnp,np=np,partial=partial,**namespace)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'actual-export-owner','exec'),env)
        return env[name]
    for input_axis in ('x','y'):
        source=NamedSharding(mesh,P(None,None,None,input_axis))
        target=NamedSharding(mesh,P(None,None,None,('x','y')))
        placed=program('_canonical_psi','place',dict(sh=target,indices=np.array([4,0,2]),physical=mu-1,carrier=mu))
        a=jax.device_put(raw,source)
        exe=placed.lower(a).compile()
        expected=raw[[4,0,2]].copy();expected[...,-1]=0
        np.testing.assert_array_equal(np.asarray(exe(a)),expected)
        for dims in re.findall(r'c128\[([0-9,]+)\]',exe.as_text()):
            assert np.prod([int(n) for n in dims.split(',')])<expected.size
    band=-(-7//(side*side))*(side*side)
    pack=program('_band_face','pack',dict(face=NamedSharding(mesh,P(None,'x','y')),
        band_carrier=band,logical_bands=7))
    exe=pack.lower(jax.device_put(raw,target)).compile()
    expected=np.pad(raw,((0,0),(0,band-7),(0,0),(0,0))).transpose(0,3,2,1).reshape(5,mu*2,band)
    np.testing.assert_array_equal(np.asarray(exe(jax.device_put(raw,target))),expected)
    for dims in re.findall(r'c128\[([0-9,]+)\]',exe.as_text()):
        assert np.prod([int(n) for n in dims.split(',')])<raw.size


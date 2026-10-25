"""The interpolation fit owns centroid columns, including uneven physical counts."""
import numpy as np
import pytest
from contextlib import nullcontext

import runtime
from runtime.source_closure import ensure_source_closure
ensure_source_closure()
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update('jax_enable_x64', True)


def _assert_local(array, expected, *, rtol=2e-11, atol=2e-11, zero_from=None):
    """Compare only addressable output tiles, also usable by the MPI proof."""
    for shard in array.addressable_shards:
        actual=np.asarray(shard.data)
        np.testing.assert_allclose(actual, expected[shard.index], rtol=rtol, atol=atol)
        assert np.isfinite(actual).all()
        if zero_from is not None:
            for axis in (-2,-1):
                idx=shard.index[axis]
                start=idx.start or 0
                keep=[slice(None)]*actual.ndim
                keep[axis]=slice(max(0,zero_from-start),None)
                np.testing.assert_array_equal(actual[tuple(keep)],0)


def check_device_placement(monkeypatch, mesh):
    """Both existing global arrays and local device seeds stay off the host."""
    from common.collectives import device_put_process_local
    host=(np.arange(8*mesh.devices.size).reshape(8,mesh.devices.size)+.25j).astype(complex)
    face=NamedSharding(mesh,P(None,('x','y')))
    global_array=device_put_process_local(host,face)
    local_array=jax.device_put(host,jax.local_devices()[0])
    original_asarray,original_array=np.asarray,np.array
    def forbid(fn):
        def call(a,*args,**kw):
            if isinstance(a,jax.Array):
                raise AssertionError('device placement attempted a host payload')
            return fn(a,*args,**kw)
        return call
    with monkeypatch.context() as guard:
        guard.setattr(np,'asarray',forbid(original_asarray))
        guard.setattr(np,'array',forbid(original_array))
        results=[device_put_process_local(a,sh) for a in (global_array,local_array)
                 for sh in (face,NamedSharding(mesh,P()))]
    for a in results:
        _assert_local(a,host,rtol=0,atol=0)
    # The host-array contract remains exact, including its scalar shape.
    _assert_local(device_put_process_local(host,face),host,rtol=0,atol=0)
    scalar=device_put_process_local(np.array(.25),NamedSharding(mesh,P()))
    assert scalar.shape==()


def test_device_placement_avoids_host_conversion_for_addressable_mesh(monkeypatch):
    if jax.device_count()<4:
        pytest.skip('requires four virtual CPU devices')
    mesh=Mesh(np.asarray(jax.devices()[:4]).reshape(2,2),('x','y'))
    check_device_placement(monkeypatch,mesh)


@pytest.mark.parametrize('side,face_route', [(2,False),(3,False),(2,True),(3,True)])
def test_clean_fit_and_evaluator_keep_only_local_centroid_columns(monkeypatch, side, face_route):
    if jax.device_count() < side**2:
        pytest.skip('requires nine virtual CPU devices')
    mesh = Mesh(np.asarray(jax.devices()[:side**2]).reshape(side, side), ('x', 'y'))
    check_clean_fit_and_evaluator(monkeypatch, mesh, face_route, reference=True)


def check_clean_fit_and_evaluator(monkeypatch, mesh, face_route, *, reference=False):
    """Shared tiny identity; a native MPI proof supplies its actual runtime mesh."""
    from bse import vq_interp as V
    from common.collectives import device_put_process_local
    from ffi import fft as F
    from runtime.padding import padded_mu_extent
    side=int(mesh.shape['x'])
    physical = 13
    mu = int(padded_mu_extent(physical, mesh))
    nq, ng = side**2, 7
    rng = np.random.default_rng(516+side)
    gs = np.array([[0,1,-1,0,0,1,-1], [0,0,0,1,-1,1,-1], [0]*ng])
    qfr = np.array([[i/side, j/side, 0.] for i in range(side) for j in range(side)])
    rmu = np.pad(rng.random((physical,3)), ((0,mu-physical),(0,0)))
    z = np.pad(rng.normal(size=(nq,physical,ng))+1j*rng.normal(size=(nq,physical,ng)),
               ((0,0),(0,mu-physical),(0,0)))
    clean_z=z.copy()
    z[:,physical:]=np.nan
    clean_rmu=rmu.copy()
    rmu[physical:]=np.nan
    c = rng.normal(size=(nq,physical,physical))+1j*rng.normal(size=(nq,physical,physical))
    c = c@c.conj().transpose(0,2,1)+np.eye(physical)
    c = np.pad(c,((0,0),(0,mu-physical),(0,mu-physical)))
    vr = 2.+rng.random((nq,ng))
    vl = .3+.2*rng.random((nq,ng))
    bvec = np.diag([.4,.4,.2])
    zx = dict(nq=nq,n_mu=mu,n_mu_logical=physical,ngkmax=ng,qfr=qfr,
              rmu_frac=rmu,bvec=bvec,celvol=100.,zeta_cutoff=50.,
              adot=np.eye(3),kgrid=np.array([side,side,1]))
    layouts=[]
    class Zeta:
        def read_q_slab(self, offset, count, *, sharding):
            layouts.append(sharding.spec)
            return device_put_process_local(z[offset:offset+count], sharding)
    zx['ZG']=Zeta()
    monkeypatch.setattr(V,'lr_gset',lambda *_a,**_k:gs)
    monkeypatch.setattr(V,'_sphere_slot',lambda _z,q,g:np.arange(ng))
    monkeypatch.setattr(V,'v_sphere_padded',lambda _z,kind=None,**_k:vl if kind else vr)
    monkeypatch.setattr(V,'v_slab_on_set',lambda _z,q,g,**_k:vl[np.argmin(np.linalg.norm(qfr-q,axis=1))])
    import distrib_la
    plan=distrib_la.plan
    monkeypatch.setattr(V,'linalg_plan',lambda op,m,**kw:plan(op,m,**dict(kw,backend='off')))
    from lxkit import memory
    admission=memory.admit
    def choose(label,choices,*a,**kw):
        if label.startswith('BSE V(q) coarse chunk'):
            return 'face' if face_route else 'q-batched'
        return admission(label,choices,*a,**kw)
    monkeypatch.setattr(memory,'admit',choose)
    face=NamedSharding(mesh,P(None,'x','y'))
    original_asarray=np.asarray
    def local_only(value,*a,**kw):
        if isinstance(value,jax.Array) and value.ndim>=2 and len(value.sharding.device_set)>1:
            raise AssertionError('production interpolation attempted a global device-to-host copy')
        return original_asarray(value,*a,**kw)
    monkeypatch.setattr(np,'asarray',local_only)
    def forbidden(*_a,**_kw):
        raise AssertionError('production interpolation attempted a global host gather')
    monkeypatch.setattr(V,'_to_host',forbidden)
    from common import collectives
    original_get=jax.device_get
    def get_local(a,*args,**kw):
        if any(isinstance(v,jax.Array) and v.ndim>=2 and len(v.sharding.device_set)>1
               for v in jax.tree.leaves(a)):
            forbidden()
        return original_get(a,*args,**kw)
    monkeypatch.setattr(jax,'device_get',get_local)
    monkeypatch.setattr(collectives,'gather_to_host',forbidden)
    with F.xla_reference() if reference else nullcontext():
        prep=V.prepare_coarse(zx,device_put_process_local(c,face),mesh,
                              q_chunk=nq,keep_host_mirrors=False,degrees={0:1})
    assert prep['S'] is prep['V_SRc_np'] is None
    assert set(prep['Fch'])=={(i*mu//side**2,(i+1)*mu//side**2) for i in range(side**2)}
    assert all(a.shape==(nq,ng,mu//side**2) for a in prep['Fch'].values())
    # Independent Tikhonov cleaning and sphere split before polynomial fitting.
    expected_f=np.zeros((nq,mu,ng),complex)
    expected_sr=np.zeros((nq,mu,mu),complex)
    for q in range(nq):
        lam,u=np.linalg.eigh(c[q])
        g=lam**2/(lam**2+(V.EPS_TIK*lam.max())**2)
        s=(u*g)@u.conj().T
        expected_f[q]=np.exp(2j*np.pi*clean_rmu@(qfr[q,:,None]+gs))*(s@clean_z[q])
        expected_sr[q]=s.conj()@((clean_z[q].conj()*(vr[q]-vl[q]))@clean_z[q].T)@s.conj()
    for (lo,hi),a in prep['Fch'].items():
        np.testing.assert_allclose(a,expected_f[:,lo:hi].transpose(0,2,1),rtol=2e-11,atol=2e-11)
        np.testing.assert_array_equal(a[...,max(physical-lo,0):] if hi>physical else a[...,:0],0)
    _assert_local(prep['V_SRc'],expected_sr,rtol=2e-10,atol=2e-10)

    # The independent polynomial system uses only physical rows, extended by zero.
    des=V.lr_design_blocks(zx,prep,degrees={0:1})
    coeff=V.fit_lr_model(des)
    packed=V.pack_coeffs(des,coeff)
    assert all(a.sharding.spec==P(None,('x','y')) for a in packed)
    ata=np.zeros((3,3));aty=np.zeros((3,mu),complex)
    for q in range(nq):
        xy=(bvec.T@(qfr[q,:,None]+gs))[:2]/(2*V.ALPHA)
        # _poly_spec(1) orders the monomials by total degree, then x power.
        phi=np.array([np.ones(ng),xy[1],xy[0]]).T
        ata+=phi.T@(vl[q,:,None]*phi)
        aty+=phi.T@(vl[q,:,None]*expected_f[q].T)
    expected_coeff=np.linalg.solve(ata+V.RIDGE*np.trace(ata)/3*np.eye(3),aty)
    _assert_local(packed[0],expected_coeff)
    assert all(block.shape[-1]==mu//side**2 for block in des['AtY'][0].values())
    for excluded in (0,nq-1):
        got=V.pack_coeffs(des,V.fit_lr_model(des,exclude=excluded))[0]
        a=des['AtA'][0][np.arange(nq)!=excluded].sum(0)
        # Concatenation is an independent, tiny test reference only.
        y=np.concatenate([des['AtY'][0][key][np.arange(nq)!=excluded].sum(0)
                          for key in sorted(des['AtY'][0])],axis=1)
        _assert_local(got,np.linalg.solve(
            a+V.RIDGE*np.trace(a)/3*np.eye(3),y),rtol=2e-11,atol=2e-11)

    target=np.array([.117,.083,0.])
    pinv=V.stencil_pinv(qfr,V.stencil_r7(zx))
    evaluator=V.make_eval_vq(zx,prep,des,mesh,mu)
    out=evaluator(target,prep['V_SRc'],pinv,packed)
    xy=(bvec.T@(target[:,None]+gs))[:2]/(2*V.ALPHA)
    phi=np.array([np.ones(ng),xy[1],xy[0]]).T
    m=(phi@expected_coeff).T
    k=bvec.T@(target[:,None]+gs);k2=(k*k).sum(0);zc=np.pi/bvec[2,2]
    weights=8*np.pi/k2*(1-np.exp(-zc*np.linalg.norm(k[:2],axis=0))*np.cos(k[2]*zc))
    weights*=np.exp(-k2/(4*V.ALPHA**2))/zx['celvol']
    a=np.exp(-2j*np.pi*clean_rmu@(target[:,None]+gs))*m*np.sqrt(weights)
    w=np.exp(-2j*np.pi*target@V.stencil_r7(zx).T)@pinv
    expected=np.einsum('q,qmn->mn',w,expected_sr)+a.conj()@a.T
    _assert_local(out,expected,rtol=2e-10,atol=2e-10,zero_from=physical)
    assert layouts
    return dict(route='face' if face_route else 'q-batched',n_mu=mu,n_mu_logical=physical,
                n_q=nq,n_G_physical=ng,
                local_form_factor_bytes=sum(a.nbytes for a in prep['Fch'].values()),
                largest_mu_block=max(hi-lo for lo,hi in prep['Fch']),
                host_global_copies_forbidden=True,
                output_sharding=str(out.sharding),coefficient_sharding=str(packed[0].sharding))


def test_interpolation_diagnostics_and_transport_refuse_before_large_materialization(monkeypatch):
    from bse import vq_interp as V
    from file_io import restart_bundle as R
    from file_io import slab_io
    monkeypatch.setattr(slab_io,'probe_availability',lambda:(False,'unit','not available'))
    with pytest.raises(ValueError,match='whole-q host replica'):
        R._zeta_mesh_for_loader(object())
    assert R._zeta_mesh_for_loader(None)==(None,False)
    huge=dict(nq=100,n_mu=100000,ngkmax=10)
    with pytest.raises(ValueError,match='fixed diagnostic limit'):
        V.prepare_coarse(huge,None,Mesh(np.asarray(jax.devices()[:1]).reshape(1,1),('x','y')))

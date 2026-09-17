"""Nonzero-frequency CT joint-projection plants, one parent per GPU."""

import argparse
import json
import os
from pathlib import Path
import subprocess


def run_checks(mesh):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import distrib_la
    from common.shard_map import shard_map
    from jax.sharding import NamedSharding, PartitionSpec as P
    from gw.shared_pole_recipe import shared_real_pole_gates_v1_r3b as gates
    from gw.shared_pole_sectors import (cross_pencil_block,
        joint_sector_pencil, reduce_sector_pencil, sector_cauchy_schwarz)

    assert jax.process_count() == 4 and mesh.size == 4
    batch = mesh.size
    spec = P(('x', 'y'))
    layout = NamedSharding(mesh, spec)
    adj = lambda a: np.swapaxes(a.conj(), -1, -2)

    def put(a):
        a = np.broadcast_to(np.asarray(a), (batch,) + np.shape(a)).copy()
        return jax.make_array_from_callback(a.shape, layout, lambda ix: a[ix])

    def mm(a, b, *, transa='N', transb='N'):
        def op(x, t):
            return x if t == 'N' else jnp.swapaxes(x.conj(), -1, -2)
        return op(a, transa) @ op(b, transb)

    native = distrib_la.plan('eigh', mesh, n=2, backend='off').native_fn

    def reduce(c, t, cross):
        return reduce_sector_pencil(joint_sector_pencil(c, t, cross, matmul=mm),
                                   eigh=native, matmul=mm, gates=gates)

    kernel = jax.jit(shard_map(reduce, mesh=mesh,
        in_specs=(spec, spec, spec), out_specs=(spec, spec), check_vma=False))
    rows = []
    # The two kept spans together span the exact two-state latent space.
    # Each diagonal sector alone has one Ritz pole; CT needs two different poles.
    yc = np.array([[1.], [0.]], complex)
    yt = np.array([[.3], [np.sqrt(.91)]], complex)
    cc = np.array([[1., .2j], [.1, .4]], complex)
    tt = np.array([[.3j, 1.], [.6, .2j], [.2, -.1j]], complex)
    metric = adj(yc) @ yt
    for ordered in (False, True):
        value = np.array([[.6, .15j], [-.15j, -.4]], complex) if ordered else np.array(
            [[2., .4j], [-.4j, 5.]], complex)
        vc, vt = (adj(yc) @ value @ yc).real.diagonal(), (adj(yt) @ value @ yt).real.diagonal()
        charge = (put(np.eye(1, dtype=complex)), put(vc), put(cc @ yc), put(tt @ yc))
        trans = (put(np.eye(1, dtype=complex)), put(vt), put(tt @ yt), put(cc @ yt))
        model, diag = kernel(charge, trans, (put(metric), put(adj(yc) @ value @ yt)))
        assert bool(jnp.all(diag['gram_valid']))
        assert bool(jnp.all(diag['retained_metric_positive']))
        c, t, lam, active = model
        errors, twins = [], []
        for z in (.7 + .2j, 1.3 + .6j, 2j):
            coordinate = z if ordered else z*z
            weights = 1 / (coordinate * lam - 1) if ordered else 1 / (coordinate-lam)
            evaluate = jax.jit(shard_map(lambda a,b,w: (a*w[:,None,:]) @ jnp.swapaxes(b.conj(),-1,-2),
                mesh=mesh, in_specs=(spec,spec,spec), out_specs=spec, check_vma=False))
            got = evaluate(c,t,weights*active)
            exact = cc @ np.linalg.solve(coordinate*value-np.eye(2) if ordered else
                                         coordinate*np.eye(2)-value, adj(tt))
            error = float(jnp.linalg.norm(got-put(exact))/jnp.linalg.norm(put(exact)))
            assert error < 1e-12, error
            errors.append(error)
            # Reuse the CC pole and its own span: a deliberately wrong twin.
            wrong = (cc @ yc) @ adj(tt @ yc) / (coordinate*vc[0]-1 if ordered else coordinate-vc[0])
            twins.append(float(np.linalg.norm(wrong-exact)/np.linalg.norm(exact)))
        assert min(twins) > .1
        rows.append(dict(name='ordered' if ordered else 'even', held_W_relative=errors,
                         CC_pole_reuse_relative=twins,
                         gram_min_relative=float(jnp.min(diag['gram_min_relative'])),
                         retained_rank=int(jnp.min(diag['retained_rank']))))

    # A physical particle-hole plant: CT is odd under z -> -z on TRS
    # data although both diagonal sectors are even. Both signed sector
    # spans have two states; their union recovers four exact signed poles.
    mu = np.array([.5, 1.25, -.5, -1.25])
    value = np.diag(mu).astype(complex)
    v = np.array([1.,1.])/np.sqrt(2)
    w = np.array([.3,np.sqrt(.91)])
    xc=np.zeros((4,2),complex); xt=np.zeros((4,2),complex)
    xc[:2,0]=v; xc[2:,1]=v
    xt[:2,0]=w; xt[2:,1]=w
    vc=np.diag(adj(xc)@value@xc).real
    vt=np.diag(adj(xt)@value@xt).real
    for broken in (False,True):
        cp=np.array([[.8,.3]],complex) + (np.array([[.1j,-.2j]]) if broken else 0)
        tp=np.array([[.2j,.7j],[.6j,-.1j],[.3j,.4j]])
        if broken:
            tp=tp+np.array([[.1,.2],[-.2,.1],[.3,-.1]])
        cfull=np.concatenate((cp,cp.conj()),axis=-1)
        tfull=np.concatenate((tp,tp.conj()),axis=-1)
        charge=(put(np.eye(2,dtype=complex)),put(vc),put(cfull@xc),put(tfull@xc))
        trans=(put(np.eye(2,dtype=complex)),put(vt),put(tfull@xt),put(cfull@xt))
        model,diag=kernel(charge,trans,(put(adj(xc)@xt),put(adj(xc)@value@xt)))
        c,t,lam,active=model
        assert bool(jnp.all(diag['gram_valid'])) and bool(jnp.all(diag['retained_metric_positive']))
        errors=[]
        for z in (.7+.2j,1.3+.6j,2j):
            exact=(cfull/(z*mu-1))@adj(tfull)
            got=evaluate(c,t,active/(z*lam-1))
            error=float(jnp.linalg.norm(got-put(exact))/jnp.linalg.norm(put(exact)))
            assert error<1e-12,error
            errors.append(error)
            if not broken:
                mirror=(cfull/(-z*mu-1))@adj(tfull)
                assert np.linalg.norm(exact+mirror)<1e-12
        # Neither exact positive pole is in the retained CC pole set.
        assert np.min(abs(1/mu[:2,None]-1/vc[None,:]))>.1
        rows.append(dict(name='physical_broken_CT' if broken else 'physical_TRS_odd_CT',
                         held_W_relative=errors,retained_rank=int(jnp.min(diag['retained_rank']))))

    # Cross Loewner entries and the confluent derivative are checked against
    # explicit latent vectors, including complex directions and rectangular CT.
    poles = np.array([.4, 1.7])
    qc = np.array([[1.], [.2j]])
    qt = np.array([[.3j], [1.], [.2]])
    def w(s):
        return (cc/(s-poles)) @ adj(tt)
    for a,b in ((.3+.4j,.8+.2j),(.3+.4j,.3-.4j)):
        derivative = -(cc/(b-poles)**2) @ adj(tt)
        def assemble(qc,qt,wa,wb,d):
            return cross_pencil_block((a,qc),(b,qt),(wa,wb,d),matmul=mm)
        block = jax.jit(shard_map(assemble,mesh=mesh,in_specs=(spec,)*5,
            out_specs=(spec,spec),check_vma=False))
        g,h = block(put(qc),put(qt),put(w(a.conjugate())),put(w(b)),put(derivative))
        xc=(adj(cc)@qc)/(a-poles)[:,None]
        xt=(adj(tt)@qt)/(b-poles)[:,None]
        error=max(float(jnp.max(jnp.abs(g-put(adj(xc)@xt)))),
                  float(jnp.max(jnp.abs(h-put(adj(xc)@(poles[:,None]*xt))))))
        assert error < 1e-12,error
        rows.append(dict(name='confluent' if b==a.conjugate() else 'cross_loewner',absolute_error=error))
    # Public endpoint components through the actual shared-pole Sigma
    # contraction, checked against an independent flattened band sum.
    from gw.mpa.sigma import synthesize_shared_pole_parents
    rng = np.random.default_rng(471)
    poles2 = np.array([[.4, .8, 1.2, 1.9]])**2
    for left_components,right_components in ((1,1),(1,3),(3,3)):
        bx = rng.normal(size=(1,2,left_components,4)) + 1j*rng.normal(size=(1,2,left_components,4))
        by = rng.normal(size=(1,2,right_components,4)) + 1j*rng.normal(size=(1,2,right_components,4))
        def face(a,spec):
            return jax.make_array_from_callback(a.shape,NamedSharding(mesh,spec),lambda ix:a[ix])
        gemm = distrib_la.gemm_plan(mesh,m=2*left_components,n=2*right_components,k=4,
                                   nq=1,dtype=np.complex128)
        def synthesize(x,y,p,i,e,t):
            return synthesize_shared_pole_parents(x,y,p,i,e,t,mesh_xy=mesh,gemm=gemm)
        synthesize = jax.jit(synthesize)
        tau=.4-.2j
        got,partner = synthesize(face(bx,P(None,'x',None,'y')),
            face(by,P(None,'y',None,'x')),jnp.asarray(poles2),
            jnp.asarray([[0,4]],jnp.int32),jnp.asarray(.1),jnp.asarray(tau))
        omega=np.sqrt(poles2)
        weights=np.exp(-1j*(omega-.1)*tau)/(2*omega)
        exact=(bx.reshape(1,-1,4)*weights[:,None]) @ adj(by.reshape(1,-1,4))
        error=float(jnp.max(jnp.abs(got-face(exact,P(None,'x','y')))))
        partner_error=float(jnp.max(jnp.abs(partner-face(exact.swapaxes(-1,-2),P(None,'x','y')))))
        assert max(error,partner_error)<1e-12,(error,partner_error)
        rows.append(dict(name=f'consumer_components_{left_components}_{right_components}',
                         absolute_error=error,partner_error=partner_error))
    def cs(c,ct,t):
        return sector_cauchy_schwarz((c,ct,t),eigh_charge=native,
                                    eigh_current=native,matmul=mm,gates=gates)
    cs=jax.jit(shard_map(cs,mesh=mesh,in_specs=(spec,)*3,out_specs=spec,check_vma=False))
    ct=np.array([[.3,0,0],[0,.4,0]],complex)
    good=cs(put(np.eye(2,dtype=complex)),put(ct),put(np.eye(3,dtype=complex)))
    red=cs(put(np.eye(2,dtype=complex)),put(3*ct),put(np.eye(3,dtype=complex)))
    good_value=float(jnp.max(good['cauchy_schwarz_squared']))
    red_value=float(jnp.max(red['cauchy_schwarz_squared']))
    assert abs(good_value-.16)<1e-12 and abs(red_value-1.44)<1e-12
    rows.append(dict(name='cauchy_schwarz',value=good_value,red_value=red_value))
    rows.extend(run_span_checks(mesh))
    assert len(rows)==11
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    from runtime import initialize_communicator_stack, finalize_process
    initialize_communicator_stack()
    import jax
    from common.collectives import resolve_mesh
    mesh=resolve_mesh()
    rows=run_checks(mesh)
    rows.extend(run_store_checks(mesh,args.output.parent))
    rows.extend(run_sigma_checks(mesh))
    assert len(rows)==19
    result=dict(status='PASS',checks=rows,expected_checks=19,
                job=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'),
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                scope='P4 sector algebra, bitwise storage and direct band-sum tau Sigma; no frequency integration or production deck')
    if jax.process_index()==0:
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    finalize_process()


def run_span_checks(mesh):
    """Original-pencil coefficient handoff and CT infinity blocks against latent states."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import distrib_la
    from common.shard_map import shard_map
    from jax.sharding import NamedSharding, PartitionSpec as P
    from gw.shared_pole_local import _mm
    from gw.shared_pole_reduction import reduce_ordered_shared_pole_pencil
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    from gw.shared_pole_sectors import ordered_cross_pencil

    spec=P(('x','y'))
    layout=NamedSharding(mesh,spec)
    adj=lambda a:a.conj().T
    rng=np.random.default_rng(472)
    signature=np.diag([1.,1.,-1.,-1.]).astype(complex)
    m=np.diag([.6,1.4,.8,1.7]).astype(complex)
    c=rng.normal(size=(2,4))+.3j*rng.normal(size=(2,4))
    t=rng.normal(size=(3,4))+.3j*rng.normal(size=(3,4))
    z=np.array([.7+.4j,1.2+.6j,-.7-.4j,-1.2-.6j])
    def family(a):
        q0=rng.normal(size=(len(a),2))+.2j*rng.normal(size=(len(a),2))
        q=np.concatenate((q0,q0),axis=1)
        qi=np.eye(len(a),dtype=complex)[:,:1]
        x=np.column_stack([np.linalg.solve(v*signature-m,adj(a)@q[:,k]) for k,v in enumerate(z)])
        k0=signature@adj(a)@qi
        full=np.concatenate((x,k0,signature@m@k0),axis=1)
        return q,qi,x,full
    qc,ic,xc,fullc=family(c)
    qt,it,xt,fullt=family(t)
    moments=[]
    power=signature@adj(t)
    for _ in range(4):
        moments.append(c@power/2)
        power=signature@m@power
    derivative=np.column_stack([-c@np.linalg.solve(v*signature-m,signature@xt[:,k]) for k,v in enumerate(z)])
    def put(a):
        a=np.broadcast_to(a,(mesh.size,)+np.shape(a)).copy()
        return jax.make_array_from_callback(a.shape,layout,lambda ix:a[ix])
    cross=jax.jit(shard_map(lambda cc,tt,actions,mom:ordered_cross_pencil(cc,tt,actions,mom,matmul=_mm),
        mesh=mesh,in_specs=(spec,)*4,out_specs=spec,check_vma=False))
    got=cross((put(z),put(qc),put(ic)),(put(z),put(qt),put(it)),
              (put(t@xc),put(c@xt),put(derivative)),tuple(put(v) for v in moments))
    expected=(adj(fullc)@signature@fullt,adj(fullc)@m@fullt,t@fullc,c@fullt)
    cross_error=max(float(jnp.max(jnp.abs(a-put(b)))) for a,b in zip(got,expected))
    assert cross_error<1e-11,cross_error
    pencil=(put(adj(fullc)@signature@fullc),put(adj(fullc)@m@fullc),put(c@fullc),put(z))
    native=distrib_la.plan('eigh',mesh,n=fullc.shape[-1],backend='off').native_fn
    def reduce(p,a):
        return reduce_ordered_shared_pole_pencil(p,a,eigh=native,matmul=_mm,gates=gates,retain_span=True)
    run=jax.jit(shard_map(reduce,mesh=mesh,in_specs=(spec,spec),out_specs=spec,check_vma=False))
    model,signed,diag,y=run(pencil,put(np.ones(fullc.shape[-1],bool)))
    assert bool(jnp.all(diag['gram_valid'])) and bool(jnp.all(diag['retained_metric_positive']))
    def check(p,y,s):
        factor,mu,active=s
        metric=_mm(y,_mm(p[1],y),transa='C')
        value=_mm(y,_mm(p[0],y),transa='C')
        eye=jnp.eye(y.shape[-1])[None]
        return (jnp.max(jnp.abs(_mm(p[2],y)-factor*active[:,None,:]),axis=(-2,-1)),
                jnp.max(jnp.abs(metric-eye*active[:,None,:]),axis=(-2,-1)),
                jnp.max(jnp.abs(value-eye*(mu*active)[:,None,:]),axis=(-2,-1)))
    check=jax.jit(shard_map(check,mesh=mesh,in_specs=(spec,)*3,out_specs=spec,check_vma=False))
    errors=[float(jnp.max(v)) for v in check(pencil,y,signed)]
    assert max(errors)<1e-10,errors
    return [dict(name='ordered_retained_span_and_cross_infinity',cross_absolute_error=cross_error,
                 output_metric_ritz_errors=errors)]


def run_store_checks(mesh,root):
    import importlib.util
    import numpy as np
    from jax.sharding import PartitionSpec as P
    from gw.shared_pole_recipe import CapacityLedger
    from file_io import shared_pole_store as store
    from file_io.slab_io import SlabIO
    spec=importlib.util.spec_from_file_location('sector_store_fixture',
        Path(__file__).resolve().parents[1]/'test_shared_pole_store.py')
    fixture=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    suffix=os.environ['SLURM_STEP_ID']
    fixture.check_roundtrip(mesh,root/f'scalar_{suffix}.h5')
    rows=[dict(name='scalar_store_roundtrip',bitwise=True)]
    meta,tables,recipe,identity=fixture._fixture(mesh)
    meta.nspinor=4
    meta.shared_pole_capacity=CapacityLedger(meta,mesh_xy=mesh)
    meta.shared_pole_capacity.reserve('fixture_live_bound',resident_bytes_per_rank=32768,
                                      workspace_bytes_per_rank=0)
    meta.shared_pole_capacity.live_stages=('fixture_live_bound',)
    canonical,_,poles,counts=fixture._model(meta)
    from common.centroid_basis import PackedCentroidBasis
    current_basis=PackedCentroidBasis.build(meta.mu_basis.canonical_indices[:4],
                                           tables['sym'],meta.fft_grid,mesh)
    from symmetry_maps import QirrTables,centroid_source_map_and_wrap
    perm,wraps=centroid_source_map_and_wrap(current_basis.canonical_indices,
        tables['sym'].sym_matrices,tables['sym'].translations,np.asarray(meta.fft_grid),extend_trs=True)
    qt=tables['qirr']
    current_tables=dict(tables,qirr=QirrTables(irr_idx_q=qt.irr_idx_q,sym_idx_q=qt.sym_idx_q,
        q_irr_frac=qt.q_irr_frac,sym_perm=perm,L_table=wraps,n_sym_spatial=qt.n_sym_spatial))
    headers={}
    for sector in ('CC','TT','CT_C','CT_T'):
        components=3 if sector in ('TT','CT_T') else 1
        basis=current_basis if components==3 else meta.mu_basis
        factor=np.concatenate([canonical[:,:basis.n_logical]*(1+1j*i)
                               for i in range(components)],axis=2)
        packed=basis.pack_host(factor,axis=1)
        path=root/f'{sector}_{suffix}.h5'
        header=store.write_shared_pole_model(path,
            fixture._device(packed,mesh,P(None,'x',None,'y')),
            fixture._device(poles,mesh,P()),counts,q_span=(0,3),meta=meta,
            tables=current_tables if components==3 else tables,recipe=recipe,
            receipts=dict(identity=identity),sector=sector,
            ordered=sector.startswith('CT'),basis=basis)
        headers[sector]=header
        store.validate_shared_pole_model(path,expected_identity=identity,mesh_xy=mesh,
                                        capacity=meta.shared_pole_capacity)
        with SlabIO(path,mode='r',mesh=mesh) as io:
            x,y,p,k=store.read_shared_pole_faces(io,(0,3),meta=meta,header=header,basis=basis)
            fixture._assert_local(x,packed)
            fixture._assert_local(y,packed)
            np.testing.assert_array_equal(np.asarray(p),poles)
            np.testing.assert_array_equal(np.asarray(k),counts)
        rows.append(dict(name=f'{sector}_store_roundtrip',bitwise=True,digest=header['digest']))
    with SlabIO(root/f'CT_C_{suffix}.h5',mode='r',mesh=mesh) as ci, \
            SlabIO(root/f'CT_T_{suffix}.h5',mode='r',mesh=mesh) as ti:
        x,y,p,k=store.read_shared_pole_cross_faces((ci,ti),(0,3),meta=meta,
            headers=(headers['CT_C'],headers['CT_T']),bases=(meta.mu_basis,current_basis))
        fixture._assert_local(x,meta.mu_basis.pack_host(canonical,axis=1))
        fixture._assert_local(y,packed)
        np.testing.assert_array_equal(np.asarray(p),poles)
        bad=dict(headers['CT_T'],K=[2,5,2])
        try:
            store.read_shared_pole_cross_faces((ci,ti),(0,3),meta=meta,
                headers=(headers['CT_C'],bad),bases=(meta.mu_basis,current_basis))
        except ValueError as error:
            assert 'disagree on K' in str(error)
        else:
            raise AssertionError('mismatched CT poles accepted')
    rows.append(dict(name='CT_endpoint_pair',bitwise=True,mismatched_census_refused=True))
    from types import SimpleNamespace
    from gw.gw_config import (refuse_headless_shared_pole_self_consistency,
                              QPSolver,HeadCorrection)
    config=SimpleNamespace(sigma=SimpleNamespace(w_model='shared_pole'),
        qp_solver=QPSolver.SELF_CONSISTENT,head=SimpleNamespace(correction=HeadCorrection.OFF),
        bispinor=True,occ_smearing_width_ry=.02)
    refuse_headless_shared_pole_self_consistency(config)
    config.bispinor=False
    try:
        refuse_headless_shared_pole_self_consistency(config)
    except ValueError as error:
        assert 'shared_pole_self_consistent_needs_a_head' in str(error)
    else:
        raise AssertionError('scalar headless SC refusal changed')
    rows.append(dict(name='headless_bispinor_metal_scope',passed=True))
    return rows


def run_sigma_checks(mesh):
    """Nonzero-time sector Sigma versus a literal band/q sum, including FD weights.

    This checks the actual FFT and band projection with distinct internal
    bare and external vertex faces. It does not certify tau quadrature.
    """
    import sys
    import jax
    import jax.numpy as jnp
    import numpy as np
    import distrib_la
    from jax.sharding import NamedSharding, PartitionSpec as P
    from gw.mpa.sigma import synthesize_shared_pole_parents, shared_pole_hole_kernel
    from gw.ppm_tau_kernel import get_shared_sigma_tau_kernel
    from gw.wavefunction_bundle import BandSlices, parent_sigma_operands, sigma_face_kernel_kwargs
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from multi_device.full_photon_head_sigma_gate import _bundle
    # Four identity-group momenta and four ports, with one C and three T
    # ports. Random spinors and unequal q factors break time reversal.
    rng=np.random.default_rng(1471)
    nk,nb,nmu,ns=4,4,4,4
    bare=rng.normal(size=(nk,nb,ns,nmu))+1j*rng.normal(size=(nk,nb,ns,nmu))
    bare*=.1
    pauli=(np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1]))
    gamma=[np.eye(4)]+[np.block([[np.zeros((2,2)),a],[a,np.zeros((2,2))]]) for a in pauli]
    external=np.stack([np.einsum('st,knt->kns',gamma[a],bare[:,:,:,a]) for a in range(4)],axis=-1)
    energy=np.linspace(.1,1.2,nk*nb).reshape(nk,nb)
    occ=1/(1+np.exp((energy-.6)/.2))
    slices=BandSlices.from_band_edges(0,0,0,nb,nb)
    wfns=_bundle(mesh,bare,energy,occ,slices)
    proj=_bundle(mesh,external,energy,occ,slices)
    xn,yr,_,_,_,_=parent_sigma_operands(wfns)
    _,_,xr,yn,_,_=parent_sigma_operands(proj)
    def put(a,spec=P()):
        a=np.asarray(a)
        return jax.make_array_from_callback(a.shape,NamedSharding(mesh,spec),lambda ix:a[ix])
    sectors=[]
    for sector,om in (('CC',np.array([.4,.9])),('TT',np.array([.6,1.1])),('CT',np.array([.7,1.3]))):
        c=np.zeros((nk,nmu,2),complex);t=np.zeros_like(c)
        if sector in ('CC','CT'):
            c[:,:1]=rng.normal(size=(nk,1,2))+1j*rng.normal(size=(nk,1,2))
        if sector in ('TT','CT'):
            t[:,1:]=rng.normal(size=(nk,3,2))+1j*rng.normal(size=(nk,3,2))
        pairs=[(c,c)] if sector=='CC' else [(t,t)] if sector=='TT' else [(c,t),(t,c)]
        for left,right in pairs:
            sectors.append((left,right,om))
    gemm=distrib_la.gemm_plan(mesh,m=nmu,n=nmu,k=2,nq=nk,dtype=np.complex128)
    faces=[(put(c[:,:,None],P(None,'x',None,'y')),put(t[:,:,None],P(None,'y',None,'x')),
            put(np.broadcast_to(om**2,(nk,2))),put(np.tile([0,2],(nk,1)).astype(np.int32)))
           for c,t,om in sectors]
    synth=jax.jit(lambda x,y,p,i,time:synthesize_shared_pole_parents(x,y,p,i,jnp.array(0.),time,
                                      mesh_xy=mesh,gemm=gemm)[0])
    hole=shared_pole_hole_kernel(mesh)
    minus=(-np.arange(nk))%nk
    def build(space,_omega,_indices,_bounds,_phase,_ref,time,_count=None):
        w=sum(synth(*f,time) for f in faces)
        return hole(w,put(minus.astype(np.int32))) if space=='val' else w
    kernel=get_shared_sigma_tau_kernel(mesh_xy=mesh,kgrid=(nk,1,1),brackets=None,
                                      w_synthesis=build,**sigma_face_kernel_kwargs(wfns))
    time=.35-.2j
    errors={}
    for space in ('cond','val'):
        e=energy if space=='cond' else -energy
        weight=1-occ if space=='cond' else occ
        actual=kernel(xn,yr,xr,yn,put(e),put(weight),space,None,
            put(np.zeros(1,np.int32)),put(np.zeros((1,6))),put(np.zeros(1,bool)),
            put(0.),put(0.),put(time))
        w=sum(np.einsum('qmp,p,qnp->qmn',c,np.exp(-1j*om*time)/(2*om),t.conj())
              for c,t,om in sectors)
        if space=='val': w=w[minus].swapaxes(-1,-2)
        expected=np.zeros((nk,nb,nb),complex)
        for k in range(nk):
            for q in range(nk):
                km=(k-q)%nk
                # Explicit transition vertices, intermediate-band sum,
                # and sector W contraction: no real-space FFT in oracle.
                v=np.einsum('ism,lsm->ilm',external[k].conj(),bare[km])
                u=np.einsum('lsn,jsn->ljn',bare[km].conj(),external[k])
                expected[k]-=np.einsum('ilm,l,mn,ljn->ij',v,
                    weight[km]*np.exp(-1j*e[km]*time),w[q],u)/nk
        error=float(jnp.max(jnp.abs(actual-put(expected,P(None,'x','y')))))
        assert error<1e-11,(space,error)
        errors[space]=error
    return [dict(name='sector_tau_sigma_direct_band_sum',absolute_errors=errors,
                 fractional_occupations=True,complex_time=[time.real,time.imag])]


if __name__=='__main__':
    main()

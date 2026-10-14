"""Exact retained OWN coordinates: complex algebra and typed physical Slab."""
from pathlib import Path
import argparse,json

def check_physical(runtime):
    import numpy as np
    import jax
    from types import SimpleNamespace
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import device_put_process_local,gather_to_host
    from common.centroid_basis import PackedCentroidBasis
    from runtime.padding import padded_axis
    from isdf.zeta_mubatch import ZStore,ZetaG,contract_v_group
    from isdf.current_coefficients import CurrentCoefficientStream
    from isdf.current_support import certify_smooth_current_support
    from isdf.atomic_breit import radial_breit_providers,_current_point_body_tile
    import isdf.atomic_breit as breit
    import sys
    sys.path.insert(0,str(Path(breit.__file__).resolve().parents[2]))
    import tests.test_streamed_current_coefficients as source_fixture
    from tests.test_support_complete_current import supported_fixture
    mesh=runtime.mesh;assert mesh.size==runtime.process_count==4
    Q,Qp,mu,na,nh,nr=3,4,8,2,4,8
    first=np.asarray([[.11,.17,.23],[.19,.31,.43],[.27,.41,.59]])
    sym=SimpleNamespace(sym_matrices=np.asarray([np.eye(3),-np.eye(3)],np.int32),translations=np.zeros((2,3)))
    basis=PackedCentroidBasis.build(np.concatenate((first,1.-first)),sym,(4,4,6),mesh,coordinate_kind='fractional')
    rng=np.random.default_rng(90371);random=lambda sh:rng.normal(size=sh)+1j*rng.normal(size=sh)
    put=lambda a,spec=P(('x','y'),None,None):device_put_process_local(np.asarray(a),NamedSharding(mesh,spec))
    host=lambda a:np.asarray(gather_to_host(a))
    zetas=[];matrices=[]
    for c in range(3):
        F=np.zeros((Qp,mu,mu),complex)
        for q in range(Q):
            # Unequal exact ranks and channel-specific columns/eigenvectors.
            count=1+(c+q)%3
            unitary=np.linalg.qr(random((6,6)))[0]
            ids=np.asarray([0,2,5])[:count]
            F[q,:6,:6][:,ids]=unitary[:,:count]*np.arange(1,count+1)[None]
        # These are outside BOTH logical axes and must not become modes.
        F[:,6:,:]=1e4-2e4j;F[:,:,6:]=3e4+4e4j;F[Q:]=np.nan
        st=ZStore(mesh=mesh,q_axis=padded_axis(Q,mesh,name='physical slab q'),mu_pad=mu,
            g_axis=padded_axis(4,4,name='initial G'),b=mu,placement='host')
        z=ZetaG(st,mesh=mesh,L_q=put(F),solver_kind='positive_rank_truncate',
            batched_route='batch_reshard',path='physical retained Slab fixture',n_rmu_solve=6,
            n_rmu=6,mu_basis=basis,ngk_per_q=np.full(Q,4),gvec_components=np.zeros((Q,3,4),np.int32))
        z.fit_vertex_mu_L=c+1;z.fit_augmentation_identity='same-field'
        z.fit_current_angular_support_identity='same-angular-support';z.fit_q_full_indices=np.asarray([0,5,1])
        zetas.append(z);matrices.append(F)
    f=SimpleNamespace(mesh=mesh,put=put,zetas=tuple(zetas),C=matrices,Q=Q,Qp=Qp,mu=mu,na=na,nh=nh,nr=nr)
    import isdf.zeta_mubatch as zm
    original=source_fixture.fixture;source_fixture.fixture=lambda:f
    original_store=zm.ZStore
    def distributed_fixture_store(**options):
        axis=options['q_axis']
        assert axis.logical==Q and axis.carrier==Qp
        options['q_axis']=padded_axis(Q,mesh,name='physical Slab fixture q')
        return original_store(**options)
    zm.ZStore=distributed_fixture_store
    try:f,wfn,plan,policy,_=supported_fixture()
    finally:
        source_fixture.fixture=original
        zm.ZStore=original_store
    centers=(wfn.atom_crys%1.)@(wfn.alat*wfn.avec);R=.2
    proof=certify_smooth_current_support(wfn,plan,f.zetas,policy,centers_cart=centers,support_radius=R)
    assert proof.physical_mixed
    radius=R*(np.arange(1,nr+1)/nr)**3;lm=np.asarray([(l,m) for l in range(2) for m in range(-l,l+1)],np.int32)
    delta=[]
    for c in range(3):
        values=random((Qp,mu,na,nh,nr))*.01
        values*=radius[None,None,None,None]**lm[None,None,None,:,0,None]*(1-(radius/R)**2)**2
        values[:,6:]=0.;values[Q:]=1e8-2e8j;delta.append(values)
    def make_banks(row_basis=None):
        stream=CurrentCoefficientStream(f.zetas,atoms=na,harmonics=nh,radial_points=nr,radial_packet=4,
            identity='same-field',angular_identity='same-angular-support',operand_kind='delta',
            smooth_support=proof,row_basis=row_basis)
        for first in (4,0):
            for c in range(3):
                packet=delta[c][...,first:first+4].reshape(Qp,mu,-1)
                stream.append(c,0,put(packet),first)
        return stream.finish()
    banks=make_banks()
    for c,z in enumerate(f.zetas):
        ng=z.gvec_components.shape[-1]
        z.store=ZStore(mesh=mesh,q_axis=padded_axis(Q,mesh,name='physical slab q'),mu_pad=mu,
            g_axis=padded_axis(ng,16,name='physical slab G'),b=mu,placement='host')
        Z=random((Q,mu,z.store.n_G))*.01;Z[:,6:]=0.
        mask=np.zeros((Q,z.store.n_G),bool)
        G=z.gvec_components
        mask[:,:ng]=(np.abs(G[:,0])<=1)&(G[:,1]==0)&(G[:,2]==0)
        mask[:,:ng]&=np.arange(ng)[None]<z.ngk_per_q[:,None]
        Z*=mask[:,None]
        z.store.write_batch(0,put(Z,P(None,('x','y'),None)))
    B=np.asarray([[1,1j,0],[1,-1j,0],[0,0,np.sqrt(2)]])/np.sqrt(2)
    provider_args=dict(body_metric='mixed_reciprocal',smooth_support=proof,
        radius=radius,lm=lm,centers_cart=centers,q_plus_G_cart=policy['q_plus_G_cart'],
        cell_volume=wfn.cell_volume,fft_points=int(np.prod(wfn.fft_grid)),support_radius=R,
        minimum_atom_image_distance=2.,reciprocal_prefactor=-8*np.pi/wfn.cell_volume,
        current_basis_rows=B,body_head_slots=policy,
        body_mask=(np.arange(ng)[None]<f.zetas[0].ngk_per_q[:,None]) & (np.sum(policy['q_plus_G_cart']**2,axis=-1)<=policy['cutoff_ry']),
        interpolation_degree=3,quadrature_order=8,fourier_points=257)
    pairs=tuple((a,b) for a in range(3) for b in range(a,3))
    tables=_current_point_body_tile(policy,np.arange(Q),0,ng,pairs,B,-8*np.pi/wfn.cell_volume,sys_dim=2)
    tables=tuple(tables[:,i] for i in range(6));keep=np.zeros((Q,2),np.int32);keep[:,1]=1
    class Writer:
        def __init__(self,z):self.mesh=z.mesh;self.path=z.path;self.payload=np.zeros((Q,mu,ng),complex)
        def write_slab(self,name,value,offset):
            first=offset[2];n=min(value.shape[-1],ng-first);self.payload[...,first:first+n]=host(value)[...,:n]
        def sync_writes(self):pass
    def run():
        writers=tuple(Writer(z) for z in f.zetas)
        V=contract_v_group(f.zetas,pairs,tables,keep=keep,zeta_ios=writers)
        return tuple(host(v) for v in V),tuple(host(z.shell) for z in f.zetas),tuple(w.payload for w in writers)
    def attach(owner,**options):
        supplied=radial_breit_providers(f.zetas,owner,**provider_args,**options)
        group=supplied[0]['group_provider'];group['compensation_tail_diagnostics']={}
        for z,p in zip(f.zetas,supplied):z.local_augmentation=p
        return group
    full=attach(banks)
    reference=run()
    full_tail=tuple(host(v) for v in full['compensation_tail_diagnostics']['weighted_fourier_K12_norm_upper'])
    from isdf.current_coordinates import select,authenticate,row_kernel
    from gw.isdf_augmentation import _price_retained_current
    from common import gpu_utils
    saved_budget=gpu_utils.device_budget_bytes
    gpu_utils.device_budget_bytes=lambda:1.
    try:assert select(f.zetas) is None
    finally:gpu_utils.device_budget_bytes=saved_budget
    selected=select(f.zetas)
    assert selected is not None and selected['rows']==4
    assert all(np.all(v[:Q,3:]==-1) for v in selected['column_ids'])
    assert all(not np.any(host(B)[:,6:]) for B in selected['thin'])
    abstract=tuple(jax.ShapeDtypeStruct((Qp,4,na*nh*nr),np.complex128,
        sharding=NamedSharding(mesh,P(('x','y'),None,None))) for _ in range(3))
    preflight=radial_breit_providers(f.zetas,abstract,_row_basis=selected,**provider_args)[0]['group_provider']
    state=dict(centers_cart=centers,lm=lm,radius=radius,sym=SimpleNamespace(irr_idx_q=np.zeros(9,int)))
    admission=_price_retained_current(f.zetas,state,preflight,selected)
    assert admission['admitted'],admission
    selected['admission']=admission
    reduced=make_banks(selected)
    group=attach(reduced)
    actual=run()
    assert group['fourier_distribution_receipt']['mode']=='mu'
    errors=[]
    for aa,bb in zip(actual,reference):
        for a,b in zip(aa,bb):
            error=float(np.max(abs(a-b)));scale=max(float(np.linalg.norm(b)),np.finfo(float).tiny)
            bound=512*np.finfo(float).eps*(na*nh*nr+ng+mu)*scale
            assert error<=bound,(error,bound)
            errors.append(dict(max_abs=error,reference_norm=scale,bound=bound))
    tail=tuple(host(v) for v in group['compensation_tail_diagnostics']['weighted_fourier_K12_norm_upper'])
    for a,b in zip(tail,full_tail):np.testing.assert_allclose(a,b,rtol=2e-11,atol=0)
    assert all(not np.any(value[:,6:]) for value in actual[2])
    proof.authenticate(f.zetas,policy,centers_cart=centers,support_radius=R)
    # A changed logical selection, source or actual OWN leaf must refuse.
    import copy
    bad=dict(selected);bad['column_ids']=list(selected['column_ids'])
    bad['column_ids'][0]=bad['column_ids'][0].copy();bad['column_ids'][0][0,0]=1
    try:authenticate(bad,f.zetas)
    except ValueError:pass
    else:raise AssertionError('changed retained column admitted')
    old=f.zetas[1].fit_q_full_indices
    f.zetas[1].fit_q_full_indices=np.asarray([0,1,5])
    try:
        try:reduced.authenticate(f.zetas)
        except ValueError:pass
        else:raise AssertionError('changed q owner admitted')
    finally:f.zetas[1].fit_q_full_indices=old
    # Logical projection ignores poisoned mu padding, exactly as solve_at_logical.
    poisoned=random((Qp,mu,3));poisoned[:,6:]=np.nan
    projected=row_kernel(mesh,Q,project=True,logical_mu=6)(selected['thin'][0],put(poisoned))
    assert np.isfinite(host(projected)).all()
    originals=tuple(z.L_q for z in f.zetas)
    try:
        for z in f.zetas:
            full=np.broadcast_to(np.eye(mu,dtype=complex),(Qp,mu,mu)).copy()
            z.L_q=put(full)
        assert select(f.zetas) is None  # six logical columns round up to M8
        f.zetas[0].solver_kind='lu'
        assert select(f.zetas) is None
        f.zetas[0].solver_kind='positive_rank_truncate'
        full[0,0,0]=np.nan;f.zetas[0].L_q=put(full)
        try:select(f.zetas)
        except ValueError:pass
        else:raise AssertionError('nonfinite logical factor admitted')
    finally:
        for z,leaf in zip(f.zetas,originals):z.L_q=leaf
        f.zetas[0].solver_kind='positive_rank_truncate'
    result=dict(errors=errors,admission=admission,distribution=group['fourier_distribution_receipt'],
        physical_mixed_support=proof.descriptor,retained_rows=4,physical_rows=8,
        scope='P4 exact selected OWN coordinates versus full apply: typed Slab, Q3 ghosts, six V, physical shell/writer, onsite, Gamma and physical tail; no material acceptance or speed claim')
    for z in f.zetas:z.close()
    return result


def check_complex_pair(runtime):
    """All six ordered four-product Grams with independent complex weights."""
    import numpy as np
    import jax.numpy as jnp
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import device_put_process_local,gather_to_host
    from isdf.zeta_mubatch import _pair_tile_kernel
    from isdf.current_coordinates import operator_kernel
    mesh=runtime.mesh;Q,Qp,M,R,G=3,4,8,4,12
    rng=np.random.default_rng(828399)
    random=lambda dims:rng.normal(size=dims)+1j*rng.normal(size=dims)
    put=lambda a,spec=P(('x','y'),None,None):device_put_process_local(np.asarray(a),NamedSharding(mesh,spec))
    host=lambda a:np.asarray(gather_to_host(a))
    B=tuple(random((Qp,M,R)) for _ in range(3))
    x,d,g=(tuple(random((Qp,R,G)) for _ in range(3)) for _ in range(3))
    pairs=tuple((a,b) for a in range(3) for b in range(a,3))
    weights=random((6,Qp,G));counts=np.asarray([G,G-2,G-1,0],np.int32)
    slots=np.tile(np.asarray([0,3],np.int32),(Qp,1))
    mask=np.arange(G)[None]<counts[:,None]
    for values in (x,d,g):
        for a in values:a*=mask[:,None]
    devB=tuple(map(put,B));tiles=tuple(tuple(map(put,v)) for v in (x,d,g))
    V=tuple(put(np.zeros((Qp,R,R),complex)) for _ in pairs)
    S=tuple(put(np.zeros((Qp,M,2),complex)) for _ in range(3))
    step=_pair_tile_kernel(mesh,G,pairs,use_local_augmentation=True,
                           return_physical=True,retained_rows=True)
    result=step(tiles[0],put(weights,P(None,('x','y'),None)),
        put(counts,P(('x','y'))),put(slots,P(('x','y'),None)),jnp.int32(0),V,S,
        (tiles[1],tiles[2]),devB)
    full=tuple(tuple(B[c]@value for c,value in enumerate(values)) for values in (x,d,g))
    expand=operator_kernel(mesh);metrics=[]
    eps=np.finfo(float).eps
    for i,(a,b) in enumerate(pairs):
        w=weights[i]*mask
        prod=lambda left,right:np.einsum('qmg,qg,qng->qmn',left.conj(),w,right)
        reference=prod(full[0][a],full[0][b])+(prod(full[0][a],full[1][b])
            +prod(full[1][a],full[0][b])+prod(full[2][a],full[2][b]))
        actual=host(expand(devB[a],result[0][i],devB[b]))
        # Bound by the four individual product operand norms, so cancellation
        # never inflates relative error by choosing a vanishing reference norm.
        norm=lambda v:float(np.linalg.norm(v))
        scale=norm(w)*(norm(full[0][a])*norm(full[0][b])
            +norm(full[0][a])*norm(full[1][b])+norm(full[1][a])*norm(full[0][b])
            +norm(full[2][a])*norm(full[2][b]))
        bound=128*eps*(G+M+R)*scale
        error=float(np.max(abs(actual-reference)))
        assert error<=bound,(i,error,bound)
        metrics.append(dict(pair=[a,b],max_abs=error,operand_bound=bound,
                            reference_norm=norm(reference)))
    for c in range(3):
        physical=full[0][c]+full[1][c]
        np.testing.assert_allclose(host(result[2][c]),physical,rtol=4e-14,atol=4e-14)
        np.testing.assert_array_equal(host(result[1][c]),host(result[2][c])[:,:,slots[0]])
    return metrics


def main():
    from runtime import initialize_communicator_stack,run_main_and_finalize
    runtime=initialize_communicator_stack()
    def execute():
        import jax
        parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
        args=parser.parse_args()
        complex_metrics=check_complex_pair(runtime)
        result=check_physical(runtime)
        result['independent_complex_pairs']=complex_metrics
        if jax.process_index()==0:
            Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result),flush=True)
        return 0
    run_main_and_finalize(execute)


if __name__=='__main__':main()

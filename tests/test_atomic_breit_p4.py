"""P4 grouped static Breit provider; independent physical references and gates."""
from pathlib import Path
import json
import os
from contextlib import contextmanager


@contextmanager
def forbid_stored_zeta_reads(zetas):
    """Prove validation runs before grouped storage access, not just arithmetic."""
    originals=[z.store.read_tile for z in zetas]
    def forbidden(*args,**kwargs):
        raise AssertionError('stored zeta read before invalid policy was rejected')
    try:
        for z in zetas:z.store.read_tile=forbidden
        yield
    finally:
        for z,method in zip(zetas,originals):z.store.read_tile=method


def check_breit_group(runtime,*,circular=True,prefactor_multiplier=-8.,
                      angular_maximum=2,field_tile=3,centroid_tile=3,compensated=False,averaged_heads=False,
                      embedded_gamma=False, publication_fixture=None,
                      mu_basis=None, canonical_to_packed_reference=None,sys_dim=3,
                      streamed_coefficients=False, price_observer=None, head_fixture_factory=None):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from scipy.linalg import lu_factor
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import device_put_process_local,gather_to_host
    from runtime.padding import padded_axis
    from isdf.zeta_mubatch import ZStore,ZetaG,contract_v_group,_require_q_owned
    from isdf.atomic_breit import radial_breit_providers
    from isdf.augmentation import local_basis_fourier
    from isdf.augmentation_breit import (current_radial_moments,
        evaluate_two_moment_compensation,static_transverse_bilinear,
        static_compensation_bilinear,curl_poisson_field_tile,compensation_curl_field_tile)

    mesh=runtime.mesh;assert mesh.size==4
    if sys_dim==2 and not (compensated and averaged_heads):
        raise ValueError('Slab witness requires compensated physical body policy')
    Q,Qp,mu,ng,gpad,gt,na,nr,R=3,4,8,7,8,4,2,8,1.1
    # Random arrays below define PACKED coordinates. The optional layout
    # witness checks file-canonical output against a literal independent map.
    reference_ids=np.arange(mu,dtype=np.int32)
    if mu_basis is not None:
        if canonical_to_packed_reference is None:
            raise ValueError('layout witness requires an independent literal reference map')
        reference_ids=np.asarray(canonical_to_packed_reference,np.int32)
        assert mu_basis.n_packed==mu_basis.n_canonical==mu
        assert reference_ids.shape==(mu_basis.n_logical,)
        assert np.array_equal(mu_basis.layout.axis.canonical_to_packed,reference_ids)
        assert np.array_equal(np.sort(reference_ids),np.flatnonzero(mu_basis.active_mask))
        assert not mu_basis.is_identity
        reference_ids=np.pad(reference_ids,(0,mu-mu_basis.n_logical))
    active_mu=np.ones(mu,bool) if mu_basis is None else np.asarray(mu_basis.active_mask)
    canonical_mu=np.ones(mu,bool) if mu_basis is None else np.arange(mu)<mu_basis.n_logical
    def canonical(value):
        return np.where(canonical_mu[None,:,None]&canonical_mu[None,None,:],
                        value[:,reference_ids][:,:,reference_ids],0.)
    volume,nfft=1000.,96;prefactor=prefactor_multiplier*np.pi/volume
    lm=np.asarray([(l,m) for l in range(angular_maximum+1) for m in range(-l,l+1)],np.int32)
    nh=len(lm);r=R*(np.arange(1,nr+1)/nr)**3
    centers=np.asarray([[.2,-.1,.3],[4.8,.2,-.2]])
    B=(np.asarray([[1,1j,0],[1,-1j,0],[0,0,np.sqrt(2)]])/np.sqrt(2)
       if circular else np.eye(3,dtype=complex))
    rng=np.random.default_rng(81922+int(circular))
    random=lambda shape:rng.normal(size=shape)+1j*rng.normal(size=shape)
    counts=np.asarray([7,5,6],np.int32);active=np.arange(ng)[None]<counts[:,None]
    K=rng.normal(size=(Q,ng,3))*1.3;K[0,0]=0.
    body=active.copy();body[2,3]=False
    head=np.zeros((Q,3,3),complex);head[0]=prefactor*np.asarray([[.7,.13j,0],[-.13j,.2,.08],[0,.08,.4]])
    head_fixture=None
    if embedded_gamma and not averaged_heads:
        raise ValueError('embedded current Gamma test requires the canonical body-head fixture')
    if averaged_heads:
        if head_fixture_factory is None:
            from tests.test_static_breit_body_heads import physical_head_fixture
            head_fixture_factory=physical_head_fixture
        head_fixture=head_fixture_factory(embedded_gamma=embedded_gamma,sys_dim=sys_dim)
        K=head_fixture['K'];body=head_fixture['body'];head=head_fixture['policy']['gamma_head_cartesian']
        assert np.array_equal(counts,head_fixture['counts'])
    gvec=(head_fixture['G'] if averaged_heads else np.zeros((Q,3,ng),np.int32))
    pairs=tuple((a,b) for a in range(3) for b in range(a,3))
    length=np.linalg.norm(K,axis=-1)
    unit=np.divide(K,length[...,None],out=np.zeros_like(K),where=length[...,None]>0)
    Pcart=np.eye(3)-unit[:,:,:,None]*unit[:,:,None,:]
    Pfit=np.einsum('ai,qgij,bj->qgab',B,Pcart,B.conj())
    Hfit=np.einsum('ai,qij,bj->qab',B,head,B.conj())
    v_all=np.divide(prefactor*Pfit,length[:,:,None,None]**2,
                    out=np.zeros_like(Pfit),where=length[:,:,None,None]>0)*body[:,:,None,None]
    v_all=np.where(length[:,:,None,None]==0,Hfit[:,None],v_all)*active[:,:,None,None]
    if averaged_heads:
        v_all=head_fixture['averaged'][:,:,None,None]*(prefactor/(8*np.pi/volume))*Pfit
        v_all=np.where(length[:,:,None,None]**2<=1e-30,Hfit[:,None],v_all)*active[:,:,None,None]
    v_tables=tuple(v_all[:,:,a,b] for a,b in pairs)
    qsh=NamedSharding(mesh,P(('x','y'),None,None))
    put=lambda value:device_put_process_local(np.asarray(value),qsh)
    rawZ=[];raw_rhs=[];smooth=[];coefficients=[];zetas=[];factors=[]
    ps_coefficients=[];ps_rhs=[];ps_rhs_host=[]
    for c in range(3):
        C=np.empty((Qp,mu,mu),complex)
        for q in range(Qp):
            F=random((mu,mu));C[q]=F.conj().T@F+(1+c)*np.eye(mu)
        C[:,~active_mu]=0.;C[:,:,~active_mu]=0.
        for index in np.flatnonzero(~active_mu):C[:,index,index]=1.
        if not circular and c==1:C=-C  # Cartesian alpha² signed LU seam.
        lu=np.stack([lu_factor(v)[0] for v in C]);piv=np.stack([lu_factor(v)[1] for v in C]).astype(np.int32)
        target=random((Qp,mu,gpad))*.04
        target[Q:]=0.;target[...,ng:]=0.;target[:,~active_mu]=0.
        Z=np.einsum('qmn,qng->qmg',C,target)
        coeff=random((Qp,mu,na,nh,nr))*(40. if sys_dim==2 else .2)
        coeff*=r[None,None,None,None]**lm[None,None,None,:,0,None]*(1-(r/R)**2)**2
        coeff[:,~active_mu]=0.
        coeff[Q:]=1e8-2e8j
        rhs=np.einsum('qmn,qnf->qmf',C,coeff.reshape(Qp,mu,-1))
        ps=random(coeff.shape)*.11
        ps*=r[None,None,None,None]**lm[None,None,None,:,0,None]*(1-(r/R)**2)**2
        ps[:,~active_mu]=0.
        ps[Q:]=-3e8+1e8j
        ps_raw=np.einsum('qmn,qnf->qmf',C,ps.reshape(Qp,mu,-1))
        ps_coefficients.append(ps);ps_rhs.append(put(ps_raw));ps_rhs_host.append(ps_raw)
        physical=target.copy()
        physical*=((np.arange(gpad)[None]<np.pad(counts,(0,Qp-Q))[:,None])[:,None])
        smooth.append(physical);coefficients.append(coeff)
        rawZ.append(Z);raw_rhs.append(put(rhs));factors.append(C)
        store=ZStore(mesh=mesh,q_axis=padded_axis(Q,4,name='static Breit q'),mu_pad=mu,
                     g_axis=padded_axis(ng,gt,name='static Breit G'),b=4,placement='host')
        for beta in range(2):
            rows=device_put_process_local(Z[:Q,beta*4:(beta+1)*4],
                NamedSharding(mesh,P(None,('x','y'),None)))
            store.write_batch(beta,rows)
        zetas.append(ZetaG(store,mesh=mesh,L_q=put(lu),
            lu_piv=device_put_process_local(piv,NamedSharding(mesh,P(('x','y'),None))),
            solver_kind='lu',batched_route='batch_reshard',n_rmu_solve=mu,n_rmu=(mu_basis.n_logical if mu_basis is not None else mu),
            mu_basis=mu_basis,ngk_per_q=counts,gvec_components=gvec,
            path='synthetic static current',print_fn=lambda line:None))
    provider_args=dict(
        smooth_rhs=tuple(ps_rhs) if compensated else None,
        body_metric="compensated" if compensated else "mixed_reciprocal",radius=r,lm=lm,centers_cart=centers,
        q_plus_G_cart=K,cell_volume=volume,fft_points=nfft,support_radius=R,
        minimum_atom_image_distance=4.5,reciprocal_prefactor=prefactor,
        current_basis_rows=B,head_cartesian=None if embedded_gamma else head,body_mask=body,
        body_head_slots=head_fixture['policy'] if averaged_heads else None,
        field_tile=field_tile,centroid_tile=centroid_tile,fourier_points=4097)
    providers=radial_breit_providers(zetas,tuple(raw_rhs),**provider_args)
    bundle=providers[0]['group_provider'];geometry=bundle['geometry'];compensation=bundle['compensation']
    radial=geometry['radial'];qr,qw=radial['quadrature_radius'],radial['quadrature_weights_dr']
    delta=np.zeros((3,Qp,mu,gpad),complex);g=np.zeros_like(delta)
    onsite=np.zeros((len(pairs),Qp,mu,mu),complex)
    cross_local=np.zeros_like(onsite)
    qprofiles=evaluate_two_moment_compensation(compensation,qr)
    # Independent host physical cardinal+Bessel transform; no provider kernels.
    for c in range(3):
        values=coefficients[c]
        for q in range(Q):
            for atom in range(na):
                sampled=np.empty((mu,nh,len(qr)),complex)
                for h,(l,m) in enumerate(lm):
                    source_map=radial['interpolation_map'].copy()
                    source_map[:radial['origin_row_count'],0]=radial['origin_factors'][l]
                    sampled[:,h]=values[q,:,atom,h]@source_map.T
                scalar=np.zeros((mu,3,nh,nr),complex);scalar[:,0]=values[q,:,atom]
                qm=current_radial_moments(scalar,geometry)[:,0]
                analytic=np.einsum('mid,idq->miq',qm,qprofiles[lm[:,0]])
                delta[c,q,:,:ng]+=local_basis_fourier(sampled,qr,qw,lm,K[q],center_cart=centers[atom])*(nfft/volume)
                g[c,q,:,:ng]+=local_basis_fourier(analytic,qr,qw,lm,K[q],center_cart=centers[atom])*(nfft/volume)
        delta[c,:Q,:,:ng]*=active[:,None];g[c,:Q,:,:ng]*=active[:,None]
    for q in range(Q):
        for atom in range(na):
            cart=[];moments=[]
            for c in range(3):
                cart.append(B[c].conj()[None,:,None,None]*coefficients[c][q,:,atom,None])
                moments.append(current_radial_moments(cart[-1],geometry))
            for index,(a,b) in enumerate(pairs):
                ed=static_transverse_bilinear(cart[a][:,None],cart[b][None],geometry,include_exterior=False)
                eg=static_compensation_bilinear(moments[a][:,None],moments[b][None],compensation,geometry,
                                                include_exterior=False)
                onsite[index,q]+=prefactor*nfft*nfft/volume*(ed-eg)
                if compensated:
                    whole=slice(0,len(qr))
                    D=[curl_poisson_field_tile(value,geometry,quadrature_slice=whole) for value in cart]
                    Gc=[compensation_curl_field_tile(value,compensation,geometry,quadrature_slice=whole) for value in moments]
                    Pc=[curl_poisson_field_tile(B[c].conj()[None,:,None,None]*ps_coefficients[c][q,:,atom,None],geometry,quadrature_slice=whole) for c in range(3)]
                    form=lambda l,r:np.einsum('mqpo,q,nqpo->mn',l.conj(),qw*qr*qr,r)
                    cross_local[index,q]+=prefactor*nfft*nfft/volume*(form(Pc[a],D[b]-Gc[b])+form(D[a]-Gc[a],Pc[b]))
    expected=[];without_cross=[];completion_terms=[]
    # Independent bulk point reference: no callback/head enters the local
    # subtraction. D_s is the actual public target body, including its head.
    bulk=np.divide(prefactor*Pfit,length[:,:,None,None]**2,
        out=np.zeros_like(Pfit),where=length[:,:,None,None]>0)*body[:,:,None,None]
    bulk*=active[:,:,None,None]
    for index,(a,b) in enumerate(pairs):
        vt=np.pad(v_tables[index],((0,1),(0,gpad-ng)))
        product=lambda left,right:np.einsum('qmg,qg,qng->qmn',left.conj(),vt,right)
        if compensated:
            base=product(smooth[a]+g[a],smooth[b]+g[b])+onsite[index]
            if sys_dim==2:
                bt=np.pad(bulk[:,:,a,b],((0,1),(0,gpad-ng)))
                bproduct=lambda left,right:np.einsum('qmg,qg,qng->qmn',left.conj(),bt,right)
                ta,tb=smooth[a]+g[a],smooth[b]+g[b]
                na0,nb0=delta[a]-g[a],delta[b]-g[b]
                subtraction=bproduct(na0,nb0)+bproduct(ta,nb0)+bproduct(na0,tb)
                # Form the hybrid via physical total samples, independently
                # of the production expansion in t/n and DeltaD.
                expected.append(product(smooth[a]+delta[a],smooth[b]+delta[b])
                                +onsite[index]+cross_local[index]-subtraction)
                dt=vt-bt
                dp=lambda left,right:np.einsum('qmg,qg,qng->qmn',left.conj(),dt,right)
                completion_terms.append((dp(na0,nb0),dp(ta,nb0),dp(na0,tb)))
            else:
                expected.append(base+cross_local[index])
        else:
            base=product(smooth[a],smooth[b])+product(g[a],g[b])+onsite[index]
            expected.append(base+product(smooth[a],delta[b])+product(delta[a],smooth[b]))
        without_cross.append(base)
    for z,p in zip(zetas,providers):z.local_augmentation=p
    if price_observer is not None:price_observer(zetas,bundle,provider_args,K)
    expected_canonical=[canonical(e) for e in expected]
    keep=np.asarray([[0,2],[0,4],[0,5]],np.int32)
    if publication_fixture is None:
        got=contract_v_group(zetas,pairs,v_tables,keep=keep,print_fn=lambda line:None)
    else:
        facts=dict(physical=tuple(np.where(canonical_mu[None,:,None],(s+d)[:Q,reference_ids,:ng],0.) for s,d in zip(smooth,delta)),
                   compensation=tuple(np.where(canonical_mu[None,:,None],(s+g0)[:Q,reference_ids,:ng],0.) for s,g0 in zip(smooth,g)),
                   expected=tuple(e[:Q] for e in expected_canonical),pairs=pairs,keep=keep,
                   expected_packed=tuple(e[:Q] for e in expected),
                   physical_packed=tuple((s+d)[:Q,:,:ng] for s,d in zip(smooth,delta)),
                   canonical_to_packed_reference=reference_ids.copy(),
                   mu_basis=mu_basis,
                   current_basis_rows=B,head_fixture=head_fixture)
        with publication_fixture['open'](zetas,facts) as writers:
            got=contract_v_group(zetas,pairs,v_tables,keep=keep,
                                 print_fn=lambda line:None,zeta_ios=writers)
        publication_fixture['verify'](zetas,got,facts)
    host=tuple(np.asarray(gather_to_host(v)) for v in got)
    error=max(float(np.max(np.abs(a-e[:Q]))) for a,e in zip(host,expected_canonical))
    normref=max(float(np.max(np.abs(e[:Q]))) for e in expected)
    assert error<2e-11,(error,normref)
    gamma_signal=0.
    if embedded_gamma:
        gamma_signal=max(float(np.max(abs(np.einsum('m,n->mn',
            (smooth[a]+delta[a])[0,:,0].conj(),(smooth[b]+delta[b])[0,:,0])*v_tables[i][0,0])))
            for i,(a,b) in enumerate(pairs))
        assert gamma_signal>1e-8,gamma_signal
    shell_error=0.
    for c,z in enumerate(zetas):
        reference=np.where(canonical_mu[None,:,None],np.take_along_axis((smooth[c]+delta[c])[:Q,reference_ids],keep[:,None],axis=2),0.)
        shell_error=max(shell_error,float(np.max(np.abs(gather_to_host(z.shell)-reference))))
    assert shell_error<2e-11,shell_error
    cross_signal=max(float(np.max(np.abs(e-b))) for e,b in zip(expected,without_cross))
    assert cross_signal>1e-5,cross_signal
    # Provider-specific q ownership and poisoned-padding controls.
    direct_coefficients=tuple(put(np.concatenate((v.reshape(Qp,mu,-1),ps.reshape(Qp,mu,-1)),axis=-1) if compensated else v.reshape(Qp,mu,-1)) for v,ps in zip(coefficients,ps_coefficients))
    direct_state=bundle['prepare'](direct_coefficients)
    assert set(direct_state)=={'coefficients','moments'}
    assert all(got is supplied for got,supplied in zip(direct_state['coefficients'],direct_coefficients))
    gamma_delta_g_error=0.
    for tile in range(2):
        ds,gs=bundle['fourier_tile'](tile,direct_state)
        for c,(d,cg) in enumerate(zip(ds,gs)):
            _require_q_owned(d,mesh,(Qp,mu,gt),name='current delta')
            _require_q_owned(cg,mesh,(Qp,mu,gt),name='current compensation')
            d,cg=map(lambda v:np.asarray(gather_to_host(v)),(d,cg))
            np.testing.assert_allclose(d,delta[c,...,tile*gt:(tile+1)*gt],atol=2e-11,rtol=2e-11)
            np.testing.assert_allclose(cg,g[c,...,tile*gt:(tile+1)*gt],atol=2e-11,rtol=2e-11)
            assert np.max(abs(d[Q:]))==0 and np.max(abs(cg[Q:]))==0
            if embedded_gamma and tile==0:
                gamma_delta_g_error=max(gamma_delta_g_error,float(np.max(abs(d[0,:,0]-cg[0,:,0]))))
                np.testing.assert_allclose(d[0,:,0],cg[0,:,0],rtol=3e-13,atol=1e-15)
    omission_errors=[]
    if sys_dim==2:
        assert bundle['completion']=='nn_tn_nt_slab_minus_bulk_point_body'
        assert bundle['bulk_reference']=='public_Bulk3D_point_body_no_head'
        assert bundle['gamma_neutral_representation_error']<2e-12
        for term in range(3):
            signal=max(float(np.max(abs(v[term]))) for v in completion_terms)
            # The exact SAME final 2e-11 absolute equality tolerance must
            # reject each independently omitted production term.
            omitted=[(e-v[term])[:,reference_ids][:,:,reference_ids] for e,v in zip(expected,completion_terms)]
            omission=max(float(np.max(abs(actual-ref[:Q]))) for actual,ref in zip(host,omitted))
            omission_errors.append(omission)
            assert omission>8*2e-11,(term,signal,omission,normref)
        changed_state=dict(direct_state,moments=tuple(1.01*m for m in direct_state['moments']))
        try:bundle['fourier_tile'](0,changed_state)
        except ValueError as exc:assert 'actual delta(0)=g(0)' in str(exc)
        else:raise AssertionError('nonneutral represented Gamma current was admitted')
    onsite_actual=bundle['onsite'](direct_state,pairs)
    onsite_error=max(float(np.max(np.abs(gather_to_host(v)-ref))) for v,ref in zip(onsite_actual,onsite+cross_local))
    assert onsite_error<2e-11,onsite_error
    with forbid_stored_zeta_reads(zetas):
        # An averaged-body bundle must refuse stale geometry/point tables BEFORE store reads.
        if averaged_heads:
            from copy import deepcopy
            from isdf.atomic_breit import _body_head_digest
            policy=head_fixture['policy'];saved_policy=deepcopy(policy)
            policy['v_avg'][1]*=1.01;policy['payload_sha256']=_body_head_digest(policy)
            try:contract_v_group(zetas,pairs,v_tables,keep=keep)
            except ValueError as exc:assert 'identity was changed' in str(exc)
            else:raise AssertionError('rehash changed averaged-head bundle was admitted')
            policy.clear();policy.update(saved_policy)
            if embedded_gamma:
                try:radial_breit_providers(zetas,tuple(raw_rhs),**dict(provider_args,head_cartesian=1.01*head))
                except ValueError as exc:assert 'Gamma tensor differs' in str(exc)
                else:raise AssertionError('explicit Gamma tensor mismatched its bound body contract')
                try:radial_breit_providers(zetas,tuple(raw_rhs),**dict(provider_args,reciprocal_prefactor=-prefactor))
                except ValueError as exc:assert 'physical signed static TT prefactor' in str(exc)
                else:raise AssertionError('a second sign/scale entered the canonical Gamma head')
                changed=dict(saved_policy)
                changed['gamma_head_cartesian']=saved_policy['gamma_head_cartesian'].copy()
                changed['gamma_head_cartesian'][0]*=1.01
                changed['payload_sha256']=_body_head_digest(changed)
                policy.clear();policy.update(changed)
                try:contract_v_group(zetas,pairs,v_tables,keep=keep)
                except ValueError as exc:assert 'identity was changed' in str(exc)
                else:raise AssertionError('rebound Gamma tensor was admitted after construction')
                policy.clear();policy.update(saved_policy)
                no_gamma=list(v_tables)
                for i in range(len(no_gamma)):
                    no_gamma[i]=no_gamma[i].copy();no_gamma[i][0,0]=0
                try:contract_v_group(zetas,pairs,no_gamma,keep=keep)
                except ValueError as exc:assert 'kernel identity' in str(exc)
                else:raise AssertionError('canonical embedded Gamma head was silently omitted')
            bare_all=head_fixture['bare'][:,:,None,None]*(prefactor/(8*np.pi/volume))*Pfit
            bare_tables=tuple(bare_all[:,:,a,b] for a,b in pairs)
            try:contract_v_group(zetas,pairs,bare_tables,keep=keep)
            except ValueError as exc:assert 'kernel identity' in str(exc)
            else:raise AssertionError('point kernel entered an averaged-body policy')
        # Every refusal is exercised before any grouped store read or contraction.
        saved=zetas[1].local_augmentation;zetas[1].local_augmentation=None
        try:contract_v_group(zetas,pairs,v_tables,keep=keep)
        except ValueError as exc:assert 'all three' in str(exc)
        else:raise AssertionError('partial current provider was admitted')
        zetas[1].local_augmentation=dict(saved,group_provider=dict(bundle))
        try:contract_v_group(zetas,pairs,v_tables,keep=keep)
        except ValueError as exc:assert 'mismatched' in str(exc)
        else:raise AssertionError('mismatched current geometry was admitted')
        zetas[1].local_augmentation=dict(saved);del zetas[1].local_augmentation['rhs']
        try:contract_v_group(zetas,pairs,v_tables,keep=keep)
        except ValueError as exc:assert 'missing' in str(exc)
        else:raise AssertionError('missing current RHS was admitted')
        zetas[1].local_augmentation=saved
        bad=list(v_tables);bad[0]=bad[0]*1.01
        try:contract_v_group(zetas,pairs,bad,keep=keep)
        except ValueError as exc:assert 'kernel identity' in str(exc)
        else:raise AssertionError('arbitrary mismatched kernel scale was admitted')
        try:contract_v_group(zetas[::-1],pairs,v_tables,keep=keep)
        except ValueError as exc:assert 'mismatched' in str(exc)
        else:raise AssertionError('current reference order was admitted')
    # No provider collective/gather: only owner-local field GEMMs and scans.
    executable=bundle['onsite_kernel'](pairs).lower(direct_state['coefficients'],direct_state['moments'],
        *bundle['onsite_kernel_arguments']).compile()
    hlo=executable.as_text().lower()
    for forbidden in ('all-gather(', 'all-to-all(', 'all-reduce('):assert forbidden not in hlo,forbidden
    memory=executable.memory_analysis()
    # Zero augmentation: exact q-layout parity, bounded face reduction error.
    for z in zetas:z.local_augmentation=None
    baseline=tuple(np.asarray(gather_to_host(v)) for v in contract_v_group(zetas,pairs,v_tables,keep=keep))
    baseline_shell=tuple(np.asarray(gather_to_host(z.shell)) for z in zetas)
    for c,(z,p) in enumerate(zip(zetas,providers)):
        zero_rhs=np.zeros((Qp,mu,na*nh*nr),complex)
        if compensated:zero_rhs=np.concatenate((zero_rhs,ps_rhs_host[c]),axis=-1)
        z.local_augmentation=dict(p,rhs=put(zero_rhs))
    zero=tuple(np.asarray(gather_to_host(v)) for v in contract_v_group(zetas,pairs,v_tables,keep=keep))
    zero_layout=bundle.get('pair_distribution_receipt',{}).get('mode','q')
    zero_diagnostics=[]
    for index,(value,reference) in enumerate(zip(zero,baseline)):
        a,b=pairs[index]
        weight=np.pad(abs(v_tables[index]),((0,1),(0,gpad-ng)))
        absolute_product=np.einsum('qmg,qg,qng->qmn',abs(smooth[a]),weight,abs(smooth[b]))
        # Two complex dot-product reductions with at most gpad terms. 64
        # accounts conservatively for real operations and both reduction trees;
        # the operand absolute-product norm, not a physical-unit tolerance,
        # scales the bound even when the result cancels.
        bound=64*np.finfo(np.float64).eps*gpad*float(np.max(absolute_product))
        difference=value-reference
        row=dict(max_abs=float(np.max(abs(difference))),rms=float(np.sqrt(np.mean(abs(difference)**2))),
            operator_norm=max(float(np.linalg.norm(v,2)) for v in difference),
            reference_max=float(np.max(abs(reference))),roundoff_bound=bound,layout=zero_layout)
        zero_diagnostics.append(row)
        if jax.process_index()==0:print('zero augmentation reduction diagnostic:',row,flush=True)
        if zero_layout=='face':assert row['max_abs']<=bound,row
        else:assert np.array_equal(value,reference),row
    assert all(np.array_equal(np.asarray(gather_to_host(z.shell)),reference)
               for z,reference in zip(zetas,baseline_shell))
    if not streamed_coefficients:
        for z in zetas:z.close()
    streamed_error = None
    streamed_aot = None
    if streamed_coefficients:
        if not compensated:
            raise ValueError('streamed fixture requires delta and PS together')
        from isdf.current_coefficients import CurrentCoefficientStream
        for c,z in enumerate(zetas,1):
            z.fit_vertex_mu_L=c
            z.fit_augmentation_identity='matched_static_current_fixture'
            z.fit_current_angular_support_identity=f'complete_L{angular_maximum}_fixture'
            z.fit_q_full_indices=np.arange(Q,dtype=np.int32)
        stream=CurrentCoefficientStream(zetas,atoms=na,harmonics=nh,
            radial_points=nr,radial_packet=4,identity='matched_static_current_fixture',
            angular_identity=f'complete_L{angular_maximum}_fixture')
        for r0 in range(0,nr,4):
            for c in range(3):
                for kind,physical in enumerate((coefficients[c],ps_coefficients[c])):
                    packet=physical[...,r0:r0+4].reshape(Qp,mu,-1)
                    rhs=np.einsum('qmn,qnf->qmf',factors[c],packet)
                    stream.append(c,kind,put(rhs),r0)
        banks=stream.finish()
        arguments=dict(provider_args);arguments.pop('smooth_rhs')
        supplied=radial_breit_providers(zetas,banks,**arguments)
        for z,p in zip(zetas,supplied):z.local_augmentation=p
        result=contract_v_group(zetas,pairs,v_tables,keep=keep)
        streamed_error=max(float(np.max(abs(gather_to_host(value)-reference)))
            for value,reference in zip(result,host))
        assert streamed_error<2e-11,streamed_error
        streamed_aot=dict(memory=stream.memory_receipt,quotes=stream.aot_quotes)
        for z in zetas:z.close()
        assert banks._banks is None
    return dict(zero_augmentation_reduction=zero_diagnostics,sys_dim=sys_dim,circular=circular,compensated=compensated,averaged_heads=averaged_heads,embedded_gamma=embedded_gamma,
        streamed_vs_incumbent_max_abs=streamed_error,streamed_solve_store_AOT=streamed_aot,
        completion_term_signals=([max(float(np.max(abs(v[t]))) for v in completion_terms) for t in range(3)] if sys_dim==2 else None),
        individually_omitted_completion_final_errors=omission_errors,
        gamma_neutral_representation_error=bundle.get('gamma_neutral_representation_error'),
        gamma_omission_signal=gamma_signal,gamma_delta_g0_error=gamma_delta_g_error,
        gamma_radial_operator_error=bundle['gamma_zero_moment_operator_error'],
        canonical_body_tie_multiplicities=head_fixture['policy']['mult'].tolist() if averaged_heads else None,prefactor=prefactor,local_scale=bundle['local_geometry_scale'],
        max_group_error=error,relative_group_error=error/normref,max_physical_shell_error=shell_error,
        max_onsite_error=onsite_error,omitted_smooth_residual_cross_signal=cross_signal,
        separate_signed_LU_factors=True,all_provider_callbacks_q_owned=True,
        poisoned_padding_inert=True,zero_exact=True,partial_mismatch_missing_rhs_kernel_order_refused=True,invalid_policy_refused_before_stored_zeta_reads=True,
        provider_no_collective_hlo=True,onsite_compiled_argument_bytes=memory.argument_size_in_bytes,
        onsite_compiled_temporary_bytes=memory.temp_size_in_bytes,
        onsite_vector_blocks=bundle['onsite_block_diagnostics'],
        coefficient_cloud_padding=bundle['coefficient_cloud_padding'],
        angular_maximum=angular_maximum,field_tile=field_tile,centroid_tile=centroid_tile,
        geometry_sha256=bundle['geometry_sha256'],fourier_cache=bundle['radial_fourier_diagnostics'])


if __name__=='__main__':
    from runtime import initialize_communicator_stack,run_main_and_finalize
    runtime=initialize_communicator_stack()
    def main():
        import jax
        from tests.test_mixed_zeta_contract import check_mixed_contract
        check_mixed_contract(runtime)  # Incumbent charge path and scalar-current refusal stay live.
        results=dict(P=4,physical_circular=check_breit_group(runtime),
                     explicit_nonphysical_scale_cartesian=check_breit_group(runtime,circular=False,prefactor_multiplier=3.1),
                     compensated_circular=check_breit_group(runtime,compensated=True),
                     compensated_cartesian=check_breit_group(runtime,circular=False,compensated=True),
                     averaged_compensated_circular=check_breit_group(runtime,compensated=True,averaged_heads=True),
                     averaged_compensated_cartesian=check_breit_group(runtime,circular=False,compensated=True,averaged_heads=True),
                     embedded_compensated_circular=check_breit_group(runtime,compensated=True,averaged_heads=True,embedded_gamma=True),
                     embedded_compensated_cartesian=check_breit_group(runtime,circular=False,compensated=True,averaged_heads=True,embedded_gamma=True),
                     slab_compensated_circular=check_breit_group(runtime,compensated=True,averaged_heads=True,embedded_gamma=True,sys_dim=2),
                     slab_compensated_cartesian=check_breit_group(runtime,circular=False,compensated=True,averaged_heads=True,embedded_gamma=True,sys_dim=2),
                     scope='Synthetic provider/group/kernel/units/layout proof with independent Cartesian smooth-neutral onsite crosses; no reconstructed-current stage or Breit GW admission.')
        if jax.process_index()==0:
            print(json.dumps(results),flush=True)
            if os.environ.get('BREIT_PROVIDER_REPORT'):
                Path(os.environ['BREIT_PROVIDER_REPORT']).write_text(json.dumps(results,indent=2)+'\n')
        return 0
    run_main_and_finalize(main)

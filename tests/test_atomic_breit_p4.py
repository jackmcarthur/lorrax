"""P4 grouped static Breit provider; independent physical references and gates."""
from pathlib import Path
import json
import os


def check_breit_group(runtime,*,circular=True,prefactor_multiplier=-8.,
                      angular_maximum=2,field_tile=3,centroid_tile=3):
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
        static_compensation_bilinear)

    mesh=runtime.mesh;assert mesh.size==4
    Q,Qp,mu,ng,gpad,gt,na,nr,R=3,4,8,7,8,4,2,8,1.1
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
    pairs=tuple((a,b) for a in range(3) for b in range(a,3))
    length=np.linalg.norm(K,axis=-1)
    unit=np.divide(K,length[...,None],out=np.zeros_like(K),where=length[...,None]>0)
    Pcart=np.eye(3)-unit[:,:,:,None]*unit[:,:,None,:]
    Pfit=np.einsum('ai,qgij,bj->qgab',B,Pcart,B.conj())
    Hfit=np.einsum('ai,qij,bj->qab',B,head,B.conj())
    v_all=np.divide(prefactor*Pfit,length[:,:,None,None]**2,
                    out=np.zeros_like(Pfit),where=length[:,:,None,None]>0)*body[:,:,None,None]
    v_all=np.where(length[:,:,None,None]==0,Hfit[:,None],v_all)*active[:,:,None,None]
    v_tables=tuple(v_all[:,:,a,b] for a,b in pairs)
    qsh=NamedSharding(mesh,P(('x','y'),None,None))
    put=lambda value:device_put_process_local(np.asarray(value),qsh)
    rawZ=[];raw_rhs=[];smooth=[];coefficients=[];zetas=[];factors=[]
    for c in range(3):
        C=np.empty((Qp,mu,mu),complex)
        for q in range(Qp):
            F=random((mu,mu));C[q]=F.conj().T@F+(1+c)*np.eye(mu)
        if not circular and c==1:C=-C  # Cartesian alpha² signed LU seam.
        lu=np.stack([lu_factor(v)[0] for v in C]);piv=np.stack([lu_factor(v)[1] for v in C]).astype(np.int32)
        target=random((Qp,mu,gpad))*.04
        target[Q:]=0.;target[...,ng:]=0.
        Z=np.einsum('qmn,qng->qmg',C,target)
        coeff=random((Qp,mu,na,nh,nr))*.2
        coeff*=r[None,None,None,None]**lm[None,None,None,:,0,None]*(1-(r/R)**2)**2
        coeff[Q:]=1e8-2e8j
        rhs=np.einsum('qmn,qnf->qmf',C,coeff.reshape(Qp,mu,-1))
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
            solver_kind='lu',batched_route='batch_reshard',n_rmu_solve=mu,n_rmu=mu,
            mu_basis=None,ngk_per_q=counts,gvec_components=np.zeros((Q,3,ng),np.int32),
            path='synthetic static current',print_fn=lambda line:None))
    providers=radial_breit_providers(zetas,tuple(raw_rhs),radius=r,lm=lm,centers_cart=centers,
        q_plus_G_cart=K,cell_volume=volume,fft_points=nfft,support_radius=R,
        minimum_atom_image_distance=4.5,reciprocal_prefactor=prefactor,
        current_basis_rows=B,head_cartesian=head,body_mask=body,
        field_tile=field_tile,centroid_tile=centroid_tile,fourier_points=4097)
    bundle=providers[0]['group_provider'];geometry=bundle['geometry'];compensation=bundle['compensation']
    radial=geometry['radial'];qr,qw=radial['quadrature_radius'],radial['quadrature_weights_dr']
    delta=np.zeros((3,Qp,mu,gpad),complex);g=np.zeros_like(delta)
    onsite=np.zeros((len(pairs),Qp,mu,mu),complex)
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
    expected=[];without_cross=[]
    for index,(a,b) in enumerate(pairs):
        vt=np.pad(v_tables[index],((0,1),(0,gpad-ng)))
        product=lambda left,right:np.einsum('qmg,qg,qng->qmn',left.conj(),vt,right)
        base=product(smooth[a],smooth[b])+product(g[a],g[b])+onsite[index]
        expected.append(base+product(smooth[a],delta[b])+product(delta[a],smooth[b]))
        without_cross.append(base)
    for z,p in zip(zetas,providers):z.local_augmentation=p
    keep=np.asarray([[0,2],[0,4],[0,5]],np.int32)
    got=contract_v_group(zetas,pairs,v_tables,keep=keep,print_fn=lambda line:None)
    host=tuple(np.asarray(gather_to_host(v)) for v in got)
    error=max(float(np.max(np.abs(a-e[:Q]))) for a,e in zip(host,expected))
    normref=max(float(np.max(np.abs(e[:Q]))) for e in expected)
    assert error<2e-11,(error,normref)
    shell_error=0.
    for c,z in enumerate(zetas):
        reference=np.take_along_axis((smooth[c]+delta[c])[:Q],keep[:,None],axis=2)
        shell_error=max(shell_error,float(np.max(np.abs(gather_to_host(z.shell)-reference))))
    assert shell_error<2e-11,shell_error
    cross_signal=max(float(np.max(np.abs(e-b))) for e,b in zip(expected,without_cross))
    assert cross_signal>1e-5,cross_signal
    # Provider-specific q ownership and poisoned-padding controls.
    direct_state=bundle['prepare'](tuple(put(v.reshape(Qp,mu,-1)) for v in coefficients))
    for tile in range(2):
        ds,gs=bundle['fourier_tile'](tile,direct_state)
        for c,(d,cg) in enumerate(zip(ds,gs)):
            _require_q_owned(d,mesh,(Qp,mu,gt),name='current delta')
            _require_q_owned(cg,mesh,(Qp,mu,gt),name='current compensation')
            d,cg=map(lambda v:np.asarray(gather_to_host(v)),(d,cg))
            np.testing.assert_allclose(d,delta[c,...,tile*gt:(tile+1)*gt],atol=2e-11,rtol=2e-11)
            np.testing.assert_allclose(cg,g[c,...,tile*gt:(tile+1)*gt],atol=2e-11,rtol=2e-11)
            assert np.max(abs(d[Q:]))==0 and np.max(abs(cg[Q:]))==0
    onsite_actual=bundle['onsite'](direct_state,pairs)
    onsite_error=max(float(np.max(np.abs(gather_to_host(v)-ref))) for v,ref in zip(onsite_actual,onsite))
    assert onsite_error<2e-11,onsite_error
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
    # Exact zero augmentation parity with the unchanged legacy grouped pass.
    for z in zetas:z.local_augmentation=None
    baseline=tuple(np.asarray(gather_to_host(v)) for v in contract_v_group(zetas,pairs,v_tables,keep=keep))
    for z,p in zip(zetas,providers):z.local_augmentation=dict(p,rhs=put(np.zeros((Qp,mu,na*nh*nr),complex)))
    zero=tuple(np.asarray(gather_to_host(v)) for v in contract_v_group(zetas,pairs,v_tables,keep=keep))
    assert all(np.array_equal(a,b) for a,b in zip(zero,baseline))
    for z in zetas:z.close()
    return dict(circular=circular,prefactor=prefactor,local_scale=bundle['local_geometry_scale'],
        max_group_error=error,relative_group_error=error/normref,max_physical_shell_error=shell_error,
        max_onsite_error=onsite_error,omitted_smooth_residual_cross_signal=cross_signal,
        separate_signed_LU_factors=True,all_provider_callbacks_q_owned=True,
        poisoned_padding_inert=True,zero_exact=True,partial_mismatch_missing_rhs_kernel_order_refused=True,
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
                     scope='Synthetic provider/group/kernel/units/layout proof; no reconstructed-current stage or Breit GW admission.')
        if jax.process_index()==0:
            print(json.dumps(results),flush=True)
            if os.environ.get('BREIT_PROVIDER_REPORT'):
                Path(os.environ['BREIT_PROVIDER_REPORT']).write_text(json.dumps(results,indent=2)+'\n')
        return 0
    run_main_and_finalize(main)

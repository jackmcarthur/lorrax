"""Complex coherent monopole enrichment, same C, physical FT and zero control."""
from pathlib import Path
from types import SimpleNamespace
import json
import os


def check_monopole_provider(runtime):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from scipy.special import spherical_jn
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from isdf.atomic_coulomb import radial_coulomb_provider, atomic_radial_metrics, _smooth_neutral_tables
    from isdf.zeta_mubatch import _local_coefficient_solve

    mesh=runtime.mesh
    assert mesh.size==4
    Q,Qp,mu,na,nr,R,ng,gt=3,4,8,2,12,1.7,7,4
    lm=np.asarray([(l,m) for l in range(3) for m in range(-l,l+1)],int);nh=len(lm);nf=na*nh*nr
    r=R*np.linspace((1e-5/R)**(1/3),1,nr)**3
    w=np.diff(np.r_[0.,(r[:-1]+r[1:])/2,R])
    rng=np.random.default_rng(610131)
    random=lambda shape:rng.normal(size=shape)+1j*rng.normal(size=shape)
    qsh=NamedSharding(mesh,P(('x','y'),None,None))
    put=lambda value:device_put_process_local(np.asarray(value),qsh)
    B=np.stack([np.linalg.qr(random((mu,mu)))[0]/np.sqrt(np.arange(1,mu+1))[None] for _ in range(Qp)])
    B[Q:]=0.
    rhs=random((Qp,mu,nf))*.004;ps=random(rhs.shape)*.01;mom=random((Qp,mu,na))*.006
    rhs[Q:]=1e7-2e7j;ps[Q:]=-3e7+1e7j;mom[Q:]=1e8+3e8j
    vectors=rng.normal(size=(Q,ng,3))*2;vectors[0,0]=0.
    ngk=np.asarray([7,5,6]);centers=np.asarray([[.1,-.3,.7],[5.7,.4,-.2]])
    volume,nfft=1728.,160
    stub=SimpleNamespace(mesh=mesh,factor=put(B),solver_kind='rank_truncate',n_rmu_solve=mu,
        ngk_per_q=ngk,store=SimpleNamespace(Q=Q,Q_pad=Qp,mu_pad=mu,g_tile=gt,g_axis=SimpleNamespace(logical=ng)))
    opts=dict(radius=r,weights_dr=w,lm=lm,centers_cart=centers,q_plus_G_cart=vectors,
        cell_volume=volume,fft_points=nfft,support_radius=R,interpolation_degree=5,quadrature_order=16)
    base=radial_coulomb_provider(stub,put(rhs),smooth_rhs=put(ps),**opts)
    enriched=radial_coulomb_provider(stub,put(rhs),smooth_rhs=put(ps),monopole_rhs=put(mom),**opts)
    assert enriched['moment_enrichment']=='served_monopole'
    solve=_local_coefficient_solve(mesh,stub.solver_kind,mu,Q)
    coefficients=solve(stub.factor,enriched['rhs'])
    host=np.asarray(gather_to_host(coefficients))
    expected=B@(B.conj().swapaxes(-2,-1)@np.concatenate((rhs,ps,mom),axis=-1));expected[Q:]=0.
    solve_error=float(np.max(abs(host-expected)));assert solve_error<2e-14
    density=host[...,:2*nf].reshape(Qp,mu,2,na,nh,nr)
    exact=host[...,2*nf:]
    tables=atomic_radial_metrics(r,w,lm[:,0],support_radius=R,fft_points=nfft,cell_volume=volume,
        interpolation_degree=5,quadrature_order=16)
    cross=_smooth_neutral_tables(tables,support_radius=R,fft_points=nfft,cell_volume=volume)
    M0=tables['moments'][0]
    interp=np.einsum('qmar,r->qma',density[:,:,0,:,0],M0);epsilon=exact-interp
    # An independent expanded physical Gram retains epsilon² in both densities;
    # its cancellation is tested against the signed provider's linear update.
    scale=2*(nfft/volume)**2;gself=scale*tables['compensation_self'][0];u=cross['smooth_compensation_cross'][0]
    reference=np.zeros((Qp,mu,mu),complex)
    delta_reference=np.zeros_like(reference)
    for atom in range(na):
        for h,(l,m) in enumerate(lm):
            row=int(l);d,s=density[:,:,0,atom,h],density[:,:,1,atom,h]
            ordinary=np.einsum('qmr,rt,qnt->qmn',d.conj(),tables['delta_metric'][row],d)
            moment=d@tables['moments'][row]
            if h==0:
                e=epsilon[:,:,atom]
                dg=d@u
                ordinary+=(e.conj()[:,:,None]*dg[:,None,:]+dg.conj()[:,:,None]*e[:,None,:]
                           +gself*e.conj()[:,:,None]*e[:,None,:])
                moment=moment+e
            comp=tables['compensation_self'][row]*scale*moment.conj()[:,:,None]*moment[:,None,:]
            delta_reference+=ordinary-comp
            s_b=np.einsum('qmr,rt,qnt->qmn',s.conj(),cross['smooth_neutral_metric'][row],d)
            reference+=ordinary-comp+s_b+s_b.swapaxes(-2,-1).conj()
    got=np.asarray(gather_to_host(enriched['onsite'](coefficients)))
    onsite_error=float(np.max(abs(got-reference)));assert onsite_error<2e-13
    omitted_epsilon=np.asarray(gather_to_host(base['onsite'](put(host[...,:2*nf]))))
    epsilon_signal=float(np.max(abs(got-omitted_epsilon)));assert epsilon_signal>1e-8
    qr,qw=tables['quadrature_radius'],tables['quadrature_weights_dr'];g=tables['compensation_quadrature_shapes'][0]
    ft_error=0.;translation_error=0.;delta_enrichment_signal=0.
    shift=np.asarray([3.,4.,-2.])
    translated=radial_coulomb_provider(stub,put(rhs),smooth_rhs=put(ps),monopole_rhs=put(mom),
                                     **dict(opts,centers_cart=centers+shift))
    for t in range(2):
        delta,comp=enriched['fourier_tile'](t,coefficients)
        old_delta,old_comp=base['fourier_tile'](t,put(host[...,:2*nf]))
        shifted_delta,shifted_comp=translated['fourier_tile'](t,coefficients)
        d,c,od,oc,sd,sc=map(lambda x:np.asarray(gather_to_host(x)),(delta,comp,old_delta,old_comp,shifted_delta,shifted_comp))
        expected_h=np.zeros_like(d)
        for q in range(Q):
            n=max(0,min(gt,int(ngk[q])-t*gt))
            K=vectors[q,t*gt:t*gt+n]
            radial=spherical_jn(0,np.linalg.norm(K,axis=-1)[:,None]*qr)@(qw*qr*qr*g)
            for atom in range(na):
                expected_h[q,:,:n]+=epsilon[q,:,atom,None]*np.sqrt(4*np.pi)*radial[None]*np.exp(-1j*K@centers[atom])[None]*(nfft/volume)
            phase=np.exp(-1j*K@shift)
            translation_error=max(translation_error,float(np.max(abs(sd[q,:,:n]-d[q,:,:n]*phase[None]))),
                                  float(np.max(abs(sc[q,:,:n]-c[q,:,:n]*phase[None]))))
        ft_error=max(ft_error,float(np.max(abs((d-od)-expected_h))),float(np.max(abs((c-oc)-expected_h))))
        delta_enrichment_signal=max(delta_enrichment_signal,float(np.max(abs(expected_h))))
        assert np.count_nonzero(d[Q:])==np.count_nonzero(c[Q:])==0
    assert ft_error<2e-13 and translation_error<2e-13
    # A known zero epsilon must retain exactly the original density arithmetic.
    zero_density=np.zeros((Qp,mu,2,na,nh,nr),complex)
    zero_density[:Q,:,0,:,0,0]=random((Q,mu,na))
    zero_density[:Q,:,1]=density[:Q,:,1]
    zero_exact=zero_density[:,:,0,:,0,0]*M0[0]
    zero_coeff=put(np.concatenate((zero_density.reshape(Qp,mu,2*nf),zero_exact),axis=-1))
    old_coeff=put(zero_density.reshape(Qp,mu,2*nf))
    zero_bitwise=np.array_equal(np.asarray(gather_to_host(enriched['onsite'](zero_coeff))),
                                np.asarray(gather_to_host(base['onsite'](old_coeff))))
    for t in range(2):
        for new,old in zip(enriched['fourier_tile'](t,zero_coeff),base['fourier_tile'](t,old_coeff)):
            zero_bitwise &= np.array_equal(np.asarray(gather_to_host(new)),np.asarray(gather_to_host(old)))
    assert zero_bitwise,'zero enrichment changed incumbent floating arithmetic'
    # The positive local-high charge model solves only delta and exact M0.
    # Its independent expanded free action must omit every PS cross term,
    # while its physical delta and compensation Fourier fields stay equal
    # to those of the paired provider at the same solved coefficients.
    delta_only=radial_coulomb_provider(stub,put(rhs),monopole_rhs=put(mom),**opts)
    delta_coefficients=solve(stub.factor,delta_only['rhs'])
    expected_delta=np.concatenate((host[...,:nf],host[...,2*nf:]),axis=-1)
    delta_solve_error=float(np.max(abs(np.asarray(gather_to_host(delta_coefficients))-expected_delta)))
    assert delta_solve_error<2e-14
    delta_free=np.asarray(gather_to_host(delta_only['onsite'](delta_coefficients)))
    delta_onsite_error=float(np.max(abs(delta_free-delta_reference)))
    assert delta_onsite_error<2e-13
    delta_fourier_error=0.
    for t in range(2):
        for delta_field,paired_field in zip(delta_only['fourier_tile'](t,delta_coefficients),
                                            enriched['fourier_tile'](t,coefficients)):
            delta_fourier_error=max(delta_fourier_error,float(np.max(abs(
                np.asarray(gather_to_host(delta_field))-np.asarray(gather_to_host(paired_field))))))
    assert delta_fourier_error<2e-13
    ps_cross_signal=float(np.max(abs(delta_free-got)))
    assert ps_cross_signal>1e-8
    try:radial_coulomb_provider(stub,put(rhs),smooth_rhs=put(ps),monopole_rhs=put(mom[...,:1]),**opts)
    except ValueError as exc:assert 'served monopole local RHS' in str(exc)
    else:raise AssertionError('moment feature shape mismatch accepted')
    result=dict(scope='Production primitive P4 coherent served-monopole proof; no physical stage or Sigma admission.',
        same_C_solution_error=solve_error,independent_complex_expanded_onsite_error=onsite_error,
        identical_delta_and_g_enrichment_FT_error=ft_error,finite_q_translation_error=translation_error,
        nonzero_enrichment_onsite_signal=epsilon_signal,nonzero_enrichment_FT_signal=delta_enrichment_signal,
        zero_epsilon_bitwise=True,q_padding_zero=True,wrong_moment_shape_refused=True,
        delta_only_same_C_solution_error=delta_solve_error,
        delta_only_expanded_free_action_error=delta_onsite_error,
        delta_only_same_physical_Fourier_error=delta_fourier_error,
        omitted_PS_cross_negative_signal=ps_cross_signal)
    if jax.process_index()==0:
        print(json.dumps(result),flush=True)
        Path(os.environ['ATOMIC_MONOPOLE_PROVIDER_REPORT']).write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    from runtime import initialize_communicator_stack,run_main_and_finalize
    runtime=initialize_communicator_stack()
    run_main_and_finalize(lambda:(check_monopole_provider(runtime),0)[1])

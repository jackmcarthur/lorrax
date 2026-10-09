"""P4 provider audit for interpolated delta and analytic compensation.

Keeps incumbent tests live; compares new Fourier tiles with direct physical
quadrature and onsite with independently pinned radial/beta field tables.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time


def check_higher_provider(runtime):
    import numpy as np
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from scipy.special import beta,betainc
    from common.collectives import device_put_process_local,gather_to_host
    from isdf.atomic_coulomb import atomic_radial_metrics,radial_coulomb_provider
    from isdf.augmentation import local_basis_fourier,radial_coulomb_metric_interpolated
    from tests.test_radial_interpolated_metric import exact_compensation_energy
    Q,Qp,mu,ng,gt,na,nr,R=3,4,10,7,4,2,32,2.6
    lm=np.array([(l,m) for l in range(5) for m in range(-l,l+1)],np.int32)
    nh=len(lm)
    r=np.geomspace(1.0638e-5,R,nr)
    # Positive legacy weights remain required even though the higher mode
    # uses its returned interpolant quadrature rather than these shell weights.
    edges=np.concatenate(([0.],(r[:-1]+r[1:])/2,[R]))
    w=np.diff(edges)
    rng=np.random.default_rng(672050)
    coefficients=(rng.normal(size=(Qp,mu,na,nh,nr))+1j*rng.normal(size=(Qp,mu,na,nh,nr)))*.005
    coefficients[Q:]=1e8-2e8j
    vectors=rng.normal(size=(Q,ng,3))*3
    vectors[0,0]=0
    vectors[1,3]=np.array([12.,.7,-.2])
    ngk=np.array([7,5,6],np.int32)
    centers=np.array([[.1,-.3,.7],[5.7,.4,-.2]])
    volume,nfft=1728.,160
    qsh=NamedSharding(runtime.mesh,P(('x','y'),None,None))
    put=lambda a:device_put_process_local(np.asarray(a),qsh)
    coeff=put(coefficients.reshape(Qp,mu,-1))
    stub=SimpleNamespace(mesh=runtime.mesh,ngk_per_q=ngk,
        store=SimpleNamespace(Q=Q,Q_pad=Qp,mu_pad=mu,g_tile=gt,g_axis=SimpleNamespace(logical=ng)))
    started=time.monotonic()
    provider=radial_coulomb_provider(stub,coeff,radius=r,weights_dr=w,lm=lm,
        centers_cart=centers,q_plus_G_cart=vectors,cell_volume=volume,fft_points=nfft,support_radius=R,
        interpolation_degree=3,quadrature_order=16,fourier_points=4097)
    provider_seconds=time.monotonic()-started
    tables=radial_coulomb_metric_interpolated(r,lm[:,0],support_radius=R,interpolation_degree=3,quadrature_order=16)
    metrics=atomic_radial_metrics(r,w,lm[:,0],support_radius=R,fft_points=nfft,cell_volume=volume,
        interpolation_degree=3,quadrature_order=16)
    qr,qw=tables['quadrature_radius'],tables['quadrature_weights_dr']
    interpolated=np.zeros((Qp,mu,na,nh,len(qr)),complex)
    compensation=np.zeros_like(interpolated)
    comp_self_errors=[];comp_field_errors=[];PSD=[]
    for row,l in enumerate(tables['degrees']):
        eigen=np.linalg.eigvalsh(metrics['delta_metric'][row])
        PSD.append(float(eigen[0]/eigen[-1]))
        analytic=exact_compensation_energy(int(l),R)
        comp_self_errors.append(float(abs(metrics['compensation_self'][row]/analytic-1)))
        # Independent analytic enclosed moment and positive-field quadrature.
        enclosed=betainc(l+1.5,7.,(qr/R)**2)
        field=4*np.pi*np.dot(qw,enclosed**2/qr**(2*l+2))+4*np.pi/(2*l+1)/R**(2*l+1)
        comp_field_errors.append(float(abs(metrics['compensation_self'][row]-field)))
        sample_map=tables['interpolation_map'].copy()
        sample_map[:tables['origin_row_count'],0]=tables['origin_factors'][row]
        g=qr**l*(1-(qr/R)**2)**6/(R**(2*l+3)*beta(l+1.5,7.)/2)
        for h in np.flatnonzero(lm[:,0]==l):
            interpolated[:,:,:,h]=coefficients[:,:,:,h]@sample_map.T
            moment=coefficients[:,:,:,h]@tables['moments'][row]
            compensation[:,:,:,h]=moment[:,:,:,None]*g
    assert min(PSD)>-1e-13,PSD
    assert max(comp_self_errors)<2e-12,comp_self_errors
    assert max(comp_field_errors)<1e-14
    refined=radial_coulomb_provider(stub,coeff,radius=r,weights_dr=w,lm=lm,
        centers_cart=centers,q_plus_G_cart=vectors,cell_volume=volume,fft_points=nfft,support_radius=R,
        interpolation_degree=3,quadrature_order=16,fourier_points=8193)
    delta_error,comp_error,padmax=0.,0.,0.
    refined_delta_error,refined_comp_error,cache_refinement_error=0.,0.,0.
    for tile in range(2):
        got,gcomp=[np.asarray(gather_to_host(x)) for x in provider['fourier_tile'](tile,coeff)]
        expected=np.zeros_like(got);expected_comp=np.zeros_like(gcomp)
        for q in range(Q):
            count=max(0,min(gt,int(ngk[q])-tile*gt))
            for a in range(na):
                k=vectors[q,tile*gt:tile*gt+count]
                if count:
                    expected[q,:,:count]+=local_basis_fourier(interpolated[q,:,a],qr,qw,lm,k,center_cart=centers[a])*(nfft/volume)
                    expected_comp[q,:,:count]+=local_basis_fourier(compensation[q,:,a],qr,qw,lm,k,center_cart=centers[a])*(nfft/volume)
        refined_got,refined_gcomp=[np.asarray(gather_to_host(x)) for x in refined['fourier_tile'](tile,coeff)]
        refined_delta_error=max(refined_delta_error,float(np.max(np.abs(refined_got-expected))))
        refined_comp_error=max(refined_comp_error,float(np.max(np.abs(refined_gcomp-expected_comp))))
        cache_refinement_error=max(cache_refinement_error,float(np.max(np.abs(refined_got-got))),float(np.max(np.abs(refined_gcomp-gcomp))))
        delta_error=max(delta_error,float(np.max(np.abs(got-expected))))
        comp_error=max(comp_error,float(np.max(np.abs(gcomp-expected_comp))))
        padmax=max(padmax,float(np.max(np.abs(got[Q:]))),float(np.max(np.abs(gcomp[Q:]))))
    assert delta_error<1e-11,delta_error
    assert comp_error<1e-11,comp_error
    assert padmax==0
    assert refined_delta_error<delta_error/8,(refined_delta_error,delta_error)
    assert refined_comp_error<comp_error/8,(refined_comp_error,comp_error)
    # Extending the certified momentum domain at unchanged spacing should
    # reproduce the same spectrum on its common interval.
    from isdf.atomic_coulomb import _radial_fourier_cache
    extended=_radial_fourier_cache(metrics,2*provider['radial_fourier_diagnostics']['maximum_wavevector'],8193)
    momenta=np.array([.01,.83,2.19,6.731,11.78])
    direct_comp=extended['compensation'](momenta)
    q_r=metrics['quadrature_radius'];q_w=metrics['quadrature_weights_dr']
    from scipy.special import spherical_jn
    exact_comp=np.stack([spherical_jn(l,momenta[:,None]*q_r)@(q_w*q_r*q_r*metrics['compensation_quadrature_shapes'][row])
                         for row,l in enumerate(metrics['degrees'])])
    extent_comp_error=float(np.max(np.abs(direct_comp-exact_comp)))
    assert extent_comp_error<2e-12,extent_comp_error
    expected_onsite=np.zeros((Qp,mu,mu),complex)
    for row,l in enumerate(tables['degrees']):
        K=tables['metric'][row]
        exactg=exact_compensation_energy(int(l),R)
        Km=np.outer(tables['moments'][row],tables['moments'][row])*exactg
        for h in np.flatnonzero(lm[:,0]==l):
            for a in range(na):
                c=coefficients[:,:,a,h]
                expected_onsite+=np.einsum('qmr,rt,qnt->qmn',c.conj(),K-Km,c)*2*(nfft/volume)**2
    got=np.asarray(gather_to_host(provider['onsite'](coeff)))
    onsite_error=float(np.max(np.abs(got[:Q]-expected_onsite[:Q])))
    assert onsite_error<2e-13,onsite_error
    clean=coefficients.copy();clean[Q:]=0
    clean_got=np.asarray(gather_to_host(provider['onsite'](put(clean.reshape(Qp,mu,-1)))))
    assert np.array_equal(got[:Q],clean_got[:Q])
    hermitian=float(np.max(np.abs(got[:Q]-got[:Q].swapaxes(-2,-1).conj())))
    assert hermitian<2e-13
    # Deliberately unresolved table must refuse before constructing kernels.
    try:
        radial_coulomb_provider(stub,coeff,radius=r,weights_dr=w,lm=lm,
            centers_cart=centers,q_plus_G_cart=vectors,cell_volume=volume,fft_points=nfft,support_radius=R,
            interpolation_degree=3,quadrature_order=16,fourier_points=9)
    except ValueError as error:
        assert 'Fourier interpolation failed' in str(error),str(error)
    else:raise AssertionError('unresolved Fourier table was accepted')
    try:
        atomic_radial_metrics(r,w,[0],support_radius=R,fft_points=nfft,cell_volume=volume,quadrature_order=16)
    except ValueError:pass
    else:raise AssertionError('field quadrature silently applied to incumbent mode')
    return dict(Q=Q,Q_pad=Qp,mu=mu,G=ng,G_pad=8,radial_samples=nr,origin_radius=float(r[0]),LM=nh,
        max_delta_fourier_error=delta_error,max_analytic_compensation_fourier_error=comp_error,
        refined_fourier_points=8193,refined_delta_fourier_error=refined_delta_error,
        refined_compensation_fourier_error=refined_comp_error,max_cache_refinement_error=cache_refinement_error,
        doubled_momentum_extent_compensation_error=extent_comp_error,
        max_onsite_error=onsite_error,max_compensation_self_relative_error=max(comp_self_errors),
        max_compensation_field_error=max(comp_field_errors),minimum_delta_metric_eigenvalue_ratio=min(PSD),
        max_padded_fourier_abs=padmax,poison_padding_does_not_move_physical_onsite=True,
        onsite_hermiticity=hermitian,unresolved_fourier_refused=True,
        provider_host_precompute_seconds=provider_seconds,fourier_cache=provider['radial_fourier_diagnostics'])


if __name__=='__main__':
    from runtime import initialize_communicator_stack,run_main_and_finalize
    runtime=initialize_communicator_stack()
    def main():
        import jax
        from tests.test_atomic_augmentation_p4 import check_radial
        assert int(runtime.mesh.size)==4
        result=dict(P=4,incumbent=check_radial(runtime),higher_provider=check_higher_provider(runtime),
            scope='P4 provider/units/padding and bounded table evidence with analytic compensation; no reconstruction fidelity or whole-fit performance claim.')
        if jax.process_index()==0:
            print(json.dumps(result),flush=True)
            out=os.environ.get('RADIAL_PROVIDER_REPORT')
            if out:Path(out).write_text(json.dumps(result,indent=2)+'\n')
        return 0
    run_main_and_finalize(main)

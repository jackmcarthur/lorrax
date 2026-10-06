"""P4 same-C compensated body, exact local cross and physical-query oracle."""
from pathlib import Path
from types import SimpleNamespace
from functools import partial
import json
import os


def check_local_cross(runtime):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from scipy.special import beta,betainc,spherical_jn
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import device_put_process_local,gather_to_host
    from common.shard_map import shard_map
    from isdf.augmentation import local_basis_fourier
    from isdf.local_rhs import local_density_rhs
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from symmetry_maps import q_negation_index
    from isdf.atomic_coulomb import (radial_coulomb_provider,atomic_radial_metrics,
                                      _smooth_neutral_tables)
    from isdf.zeta_mubatch import _local_coefficient_solve
    assert runtime.mesh.size==4
    mesh=runtime.mesh
    Q,Qp,mu,ng,gt,na,nr,R=3,4,8,7,4,2,12,1.7
    lm=np.asarray([(l,m) for l in range(3) for m in range(-l,l+1)],np.int32)
    nh=len(lm)
    r=R*np.linspace((1e-5/R)**(1/3),1,nr)**3
    edges=np.concatenate(([0.],(r[:-1]+r[1:])/2,[R]));w=np.diff(edges)
    rng=np.random.default_rng(610101)
    random=lambda shape:rng.normal(size=shape)+1j*rng.normal(size=shape)
    qsh=NamedSharding(mesh,P(('x','y'),None,None))
    put=lambda value:device_put_process_local(np.asarray(value),qsh)
    B=np.stack([np.linalg.qr(random((mu,mu)))[0]/np.sqrt(np.arange(1,mu+1))[None]
                for _ in range(Qp)])
    B[Q:]=0.
    drhs=random((Qp,mu,na*nh*nr))*.004
    srhs=random(drhs.shape)*.01
    drhs[Q:]=1e7-2e7j;srhs[Q:]=-3e7+1e7j
    vectors=rng.normal(size=(Q,ng,3))*2.;vectors[0,0]=0.
    ngk=np.array([7,5,6],np.int32)
    centers=np.array([[.1,-.3,.7],[5.7,.4,-.2]])
    volume,nfft=1728.,160
    stub=SimpleNamespace(mesh=mesh,factor=put(B),solver_kind='rank_truncate',
        n_rmu_solve=mu,ngk_per_q=ngk,
        store=SimpleNamespace(Q=Q,Q_pad=Qp,mu_pad=mu,g_tile=gt,g_axis=SimpleNamespace(logical=ng)))
    provider=radial_coulomb_provider(stub,put(drhs),smooth_rhs=put(srhs),radius=r,weights_dr=w,lm=lm,
        centers_cart=centers,q_plus_G_cart=vectors,cell_volume=volume,fft_points=nfft,
        support_radius=R,interpolation_degree=5,quadrature_order=16)
    assert provider['compensated_body'] and provider['body_metric']=='compensated'
    assert provider['local_cross_policy']=='onsite_smooth_neutral'
    coeff=_local_coefficient_solve(mesh,stub.solver_kind,mu,Q)(stub.factor,provider['rhs'])
    host=np.asarray(gather_to_host(coeff)).reshape(Qp,mu,2,na,nh,nr)
    expected=(B@(B.conj().swapaxes(-2,-1)@np.concatenate((drhs,srhs),axis=-1))).reshape(host.shape)
    expected[Q:]=0.
    solve_error=float(np.max(abs(host-expected)))
    assert solve_error<2e-14,solve_error
    tables=atomic_radial_metrics(r,w,lm[:,0],support_radius=R,fft_points=nfft,
        cell_volume=volume,interpolation_degree=5,quadrature_order=16)
    tables.update(_smooth_neutral_tables(tables,support_radius=R,fft_points=nfft,cell_volume=volume))
    qr,qw=tables['quadrature_radius'],tables['quadrature_weights_dr']
    delta=np.zeros((Qp,mu,na,nh,len(qr)),complex)
    comp=np.zeros_like(delta)
    max_cross_metric_error=0.
    # Independent Legendre antiderivatives reconstruct every cardinal
    # enclosed moment. Positive field cross gives <f_i|v|g_l> without the
    # analytic g potential used in provider.py.
    x,wx=np.polynomial.legendre.leggauss(16)
    scale=2*(nfft/volume)**2
    for row,l in enumerate(tables['degrees']):
        sample=tables['interpolation_map'].copy()
        sample[:16,0]=tables['origin_factors'][row]
        prefix=np.zeros(nr);M=np.zeros((len(qr),nr))
        for start in range(0,len(qr),16):
            rr=qr[start:start+16]
            half=(rr[-1]-rr[0])/(x[-1]-x[0])
            poly=np.polynomial.legendre.legfit(x,sample[start:start+16]*rr[:,None]**(l+2),2*int(l)+7)
            primitive=np.polynomial.legendre.legint(poly,axis=0)
            base=np.polynomial.legendre.legval(-1.,primitive)
            values=np.polynomial.legendre.legval(x,primitive).T
            M[start:start+16]=prefix+half*(values-base)
            prefix+=half*(np.polynomial.legendre.legval(1.,primitive)-base)
        gM=betainc(l+1.5,7.,(qr/R)**2)
        independent_u=scale*(4*np.pi*(qw*gM/qr**(2*l+2))@M
            +4*np.pi/(2*l+1)*prefix/R**(2*l+1))
        max_cross_metric_error=max(max_cross_metric_error,
            float(np.max(abs(independent_u-tables['smooth_compensation_cross'][row]))))
        g=qr**l*(1-(qr/R)**2)**6/(R**(2*l+3)*beta(l+1.5,7)/2)
        for h in np.flatnonzero(lm[:,0]==l):
            delta[:,:,:,h]=host[:,:,0,:,h]@sample.T
            moment=host[:,:,0,:,h]@tables['moments'][row]
            comp[:,:,:,h]=moment[:,:,:,None]*g
    assert max_cross_metric_error<2e-12,max_cross_metric_error
    # True compensation body Fourier tiles versus direct physical quadrature.
    comp_error=0.;comp_pad=0.
    ngpad=8
    comp_all=np.zeros((Qp,mu,ngpad),complex)
    delta_all=np.zeros_like(comp_all)
    for tile in range(2):
        dft,compft=provider['fourier_tile'](tile,coeff)
        actual=np.asarray(gather_to_host(compft))
        delta_all[...,tile*gt:(tile+1)*gt]=np.asarray(gather_to_host(dft))
        reference=np.zeros_like(actual)
        for q in range(Q):
            count=max(0,min(gt,int(ngk[q])-tile*gt))
            if count:
                for atom in range(na):
                    reference[q,:,:count]+=local_basis_fourier(comp[q,:,atom],qr,qw,lm,
                        vectors[q,tile*gt:tile*gt+count],center_cart=centers[atom])*(nfft/volume)
        comp_error=max(comp_error,float(np.max(abs(actual-reference))))
        comp_pad=max(comp_pad,float(np.max(abs(actual[Q:]))))
        comp_all[...,tile*gt:(tile+1)*gt]=actual
    assert comp_error<2e-13,comp_error
    assert comp_pad==0.
    # Physical head and writer queries are delta, never compensation.
    physical=delta_all[...,:ng]
    physical_reference=np.zeros_like(physical)
    for q in range(Q):
        for atom in range(na):
            physical_reference[q,:,:ngk[q]]+=local_basis_fourier(delta[q,:,atom],qr,qw,lm,
                vectors[q,:ngk[q]],center_cart=centers[atom])*(nfft/volume)
    physical_error=float(np.max(abs(physical-physical_reference)))
    wrong_density_signal=float(np.max(abs(physical-comp_all[...,:ng])))
    assert physical_error<2e-13,physical_error
    assert wrong_density_signal>1e-5,wrong_density_signal
    onsite=np.asarray(gather_to_host(provider['onsite'](coeff)))
    reference=np.zeros_like(onsite)
    omitted=np.zeros_like(onsite)
    for h,(l,m) in enumerate(lm):
        row=np.flatnonzero(tables['degrees']==l)[0]
        for atom in range(na):
            d,s=host[:,:,0,atom,h],host[:,:,1,atom,h]
            base=np.einsum('qmr,rt,qnt->qmn',d.conj(),
                tables['delta_metric'][row]-tables['compensation_metric'][row],d)
            cross=np.einsum('qmr,rt,qnt->qmn',s.conj(),tables['smooth_neutral_metric'][row],d)
            reference+=base+cross+cross.swapaxes(-2,-1).conj();omitted+=base
    onsite_error=float(np.max(abs(onsite-reference)))
    onsite_cross_signal=float(np.max(abs(reference-omitted)))
    assert onsite_error<2e-13,onsite_error
    assert onsite_cross_signal>1e-6,onsite_cross_signal
    assert np.max(abs(onsite[Q:]))==0.
    # Rank-local body contraction and source ownership have no collectives.
    @jax.jit
    @partial(shard_map,mesh=mesh,in_specs=(P(('x','y'),None,None),P(('x','y'),None,None),
                                  P(('x','y'),None)),out_specs=qsh.spec,check_vma=False)
    def body(s,g,v):
        return jnp.einsum('qmg,qg,qng->qmn',(s+g).conj(),v,s+g)
    smooth=random((Qp,mu,ngpad))*.03;smooth[Q:]=0.
    mask=np.zeros((Qp,ngpad),bool);mask[:Q]=np.arange(ngpad)[None]<ngk[:,None]
    smooth*=mask[:,None]
    vt=rng.uniform(.2,1.1,size=(Qp,ngpad))*mask
    body_value=np.asarray(gather_to_host(body(put(smooth),put(comp_all),
        device_put_process_local(vt,NamedSharding(mesh,P(('x','y'),None))))))
    body_ref=np.einsum('qmg,qg,qng->qmn',(smooth+comp_all).conj(),vt,smooth+comp_all)
    body_error=float(np.max(abs(body_value-body_ref)))
    assert body_error<2e-13,body_error
    hlo=str(body.lower(put(smooth),put(comp_all),
        device_put_process_local(vt,NamedSharding(mesh,P(('x','y'),None)))).compiler_ir(dialect='hlo').as_hlo_text())
    assert not any(name in hlo.lower() for name in ('all-gather','all-reduce','collective-permute'))
    # Refuse the old face layout rather than silently resharding it.
    bad=device_put_process_local(drhs,NamedSharding(mesh,P(None,'x','y')))
    try:
        radial_coulomb_provider(stub,bad,smooth_rhs=put(srhs),radius=r,weights_dr=w,lm=lm,
            centers_cart=centers,q_plus_G_cart=vectors,cell_volume=volume,
            fft_points=nfft,support_radius=R)
    except ValueError as exc:
        assert 'q-owner' in str(exc) or 'delta local RHS' in str(exc)
    else:raise AssertionError('non-q-owned RHS accepted')
    geometry=dict(radius=r,weights_dr=w,lm=lm,centers_cart=centers,
        q_plus_G_cart=vectors,cell_volume=volume,fft_points=nfft,support_radius=R)
    try:
        radial_coulomb_provider(stub,put(drhs),smooth_rhs=put(srhs[...,:-1]),
                               interpolation_degree=5,**geometry)
    except ValueError as exc:
        assert 'smooth PS local RHS' in str(exc)
    else:raise AssertionError('mismatched smooth RHS feature axis accepted')
    try:
        radial_coulomb_provider(stub,put(drhs),smooth_rhs=put(srhs),**geometry)
    except ValueError as exc:
        assert 'physical density interpolation' in str(exc)
    else:raise AssertionError('onsite cross silently used shell collocation')
    current=SimpleNamespace(**dict(vars(stub),solver_kind='lu'))
    try:
        radial_coulomb_provider(current,put(drhs),smooth_rhs=put(srhs),
                               interpolation_degree=5,**geometry)
    except ValueError as exc:
        assert 'scalar charge factor' in str(exc)
    else:raise AssertionError('current factor silently used scalar cross policy')
    result=dict(scope='Production primitive P4 proof; no physical stage or Sigma admission.',
        P=4,Q=Q,Q_pad=Qp,mu=mu,Nrad=nr,LM=nh,atoms=na,
        same_C_paired_solution_error=solve_error,
        independent_positive_field_cross_row_error=max_cross_metric_error,
        analytic_compensation_FT_error=comp_error,physical_delta_query_error=physical_error,
        physical_compensation_difference=wrong_density_signal,
        complete_onsite_error=onsite_error,omitted_smooth_neutral_signal=onsite_cross_signal,
        q_padding_exact_zero=True,face_layout_refused=True,compensated_body_error=body_error,
        wrong_PS_shape_refused=True,shell_collocation_refused=True,current_factor_refused=True,
        HLO_body_collectives=False,
        limitation='Positive field interpolation model and synthetic same-C RHS. Actual PS producer covariance, complete physical reference and whole-fit writer-on timing remain required.')
    if jax.process_index()==0:
        print(json.dumps(result),flush=True)
        output=os.environ.get('ATOMIC_LOCAL_CROSS_REPORT')
        if output:Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return 0


if __name__=='__main__':
    from runtime import initialize_communicator_stack,run_main_and_finalize
    runtime=initialize_communicator_stack()
    def main():
        return check_local_cross(runtime)
    run_main_and_finalize(main)

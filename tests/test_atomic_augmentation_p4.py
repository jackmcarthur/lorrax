"""Independent P4 radial-provider, projection and corrected-sample audit.

Run ``python -m tests.test_atomic_augmentation_p4`` through one P4 lx pool
leg. Synthetic atomic data test the mathematical/distributed seams; they do
not authenticate an atomic sidecar or establish core-recovery accuracy.
"""
from fractions import Fraction
from itertools import product
import json
from math import comb
import os
from pathlib import Path
from types import SimpleNamespace


def check_radial(runtime):
    import numpy as np
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from isdf.atomic_coulomb import atomic_radial_metrics, radial_coulomb_provider
    from isdf.augmentation import local_basis_fourier

    mesh = runtime.mesh
    Q, Qp, mu, ng, gt, na, nr, R = 3, 4, 10, 5, 4, 2, 40, 2.3
    lm = np.array([(l,m) for l in range(3) for m in range(-l,l+1)],np.int32)
    nh = len(lm)
    nodes, weights = np.polynomial.legendre.leggauss(nr)
    r,w = R*(nodes+1)/2,R*weights/2
    centers = np.array([[.2,-.4,.6],[5.2,.1,-.7]])
    rng = np.random.default_rng(691209)
    random = lambda shape: rng.normal(size=shape)+1j*rng.normal(size=shape)
    coefficients = random((Qp,mu,na,nh,nr))*.03
    coefficients[Q:] = (3e7-2e7j)
    kg = rng.normal(size=(Q,ng,3))*.8
    kg[0,0] = 0
    ngk = np.array([5,3,4],np.int32)
    volume, nfft = 1000.,96
    qsh = NamedSharding(mesh,P(('x','y'),None,None))
    put = lambda value: device_put_process_local(np.asarray(value),qsh)
    stub = SimpleNamespace(mesh=mesh,ngk_per_q=ngk,
        store=SimpleNamespace(Q=Q,Q_pad=Qp,mu_pad=mu,g_tile=gt,g_axis=SimpleNamespace(logical=ng)))
    coeff = put(coefficients.reshape(Qp,mu,-1))
    provider = radial_coulomb_provider(stub,coeff,radius=r,weights_dr=w,lm=lm,
        centers_cart=centers,q_plus_G_cart=kg,cell_volume=volume,fft_points=nfft,support_radius=R)
    assert provider['rhs'] is coeff
    # Explicitly certify the synthetic geometry in a cubic 10-bohr cell.
    image_offsets=np.array(list(product(range(-1,2),repeat=3)))*10
    distances=np.linalg.norm(centers[1]-centers[0]+image_offsets,axis=1)
    assert 2*R < np.min(distances) and 2*R < 10
    bad_sets=[lm.astype(float),lm.copy(),lm[:-1]]
    bad_sets[0][0,0]=.5
    bad_sets[1][2]=bad_sets[1][1]
    for bad in bad_sets:
        bad_rhs=put(np.zeros((Qp,mu,na*len(bad)*nr),complex))
        try:
            radial_coulomb_provider(stub,bad_rhs,radius=r,weights_dr=w,lm=bad,
                centers_cart=centers,q_plus_G_cart=kg,cell_volume=volume,fft_points=nfft,support_radius=R)
        except ValueError:
            pass
        else:
            raise AssertionError(f'fractional/duplicate/incomplete LM set accepted: {bad.tolist()}')
    shape = np.stack([r**l*(1-(r/R)**2)**6 for l in range(3)])
    moment = np.stack([w*r**(l+2) for l in range(3)])
    shape /= np.einsum('lr,lr->l',shape,moment)[:,None]
    compensated = np.zeros_like(coefficients)
    for h,(l,m) in enumerate(lm):
        moments = np.einsum('qmar,r->qma',coefficients[:,:,:,h],moment[l])
        compensated[:,:,:,h] = moments[:,:,:,None]*shape[l]
    fourier_error, compensation_error, padded_max = 0.,0.,0.
    for tile in range(2):
        delta, comp = [np.asarray(gather_to_host(value)) for value in provider['fourier_tile'](tile,coeff)]
        expected_delta,expected_comp = np.zeros_like(delta),np.zeros_like(comp)
        for q in range(Q):
            count = max(0,min(gt,int(ngk[q])-tile*gt))
            for atom in range(na):
                vectors = kg[q,tile*gt:tile*gt+count]
                if count:
                    expected_delta[q,:,:count] += local_basis_fourier(coefficients[q,:,atom],r,w,lm,vectors,center_cart=centers[atom])*(nfft/volume)
                    expected_comp[q,:,:count] += local_basis_fourier(compensated[q,:,atom],r,w,lm,vectors,center_cart=centers[atom])*(nfft/volume)
        fourier_error=max(fourier_error,float(np.max(np.abs(delta-expected_delta))))
        compensation_error=max(compensation_error,float(np.max(np.abs(comp-expected_comp))))
        padded_max=max(padded_max,float(np.max(np.abs(delta[Q:]))),float(np.max(np.abs(comp[Q:]))))
    assert fourier_error < 2e-13,(fourier_error,compensation_error)
    assert compensation_error < 2e-13
    assert padded_max == 0

    # Independent coupled-LM double integration: dense min/max kernel,
    # not the producer's radial Poisson scan or precomputed metrics.
    expected_onsite=np.zeros((Qp,mu,mu),complex)
    minimum=np.minimum(r[:,None],r[None,:]);maximum=np.maximum(r[:,None],r[None,:])
    measure=w*r*r
    for h,(l,m) in enumerate(lm):
        kernel=(4*np.pi/(2*l+1))*minimum**l/maximum**(l+1)*measure[:,None]*measure[None]
        for atom in range(na):
            c,g=coefficients[:,:,atom,h],compensated[:,:,atom,h]
            expected_onsite += np.einsum('qmr,rt,qnt->qmn',c.conj(),kernel,c)-np.einsum('qmr,rt,qnt->qmn',g.conj(),kernel,g)
    expected_onsite *= 2*(nfft/volume)**2
    onsite=np.asarray(gather_to_host(provider['onsite'](coeff)))
    onsite_error=float(np.max(np.abs(onsite[:Q]-expected_onsite[:Q])))
    assert onsite_error < 3e-13,onsite_error
    hermitian=float(np.max(np.abs(onsite[:Q]-onsite[:Q].swapaxes(-2,-1).conj())))
    assert hermitian < 3e-13,hermitian
    clean=coefficients.copy();clean[Q:]=0
    clean_onsite=np.asarray(gather_to_host(provider['onsite'](put(clean.reshape(Qp,mu,-1)))))
    assert np.array_equal(onsite[:Q],clean_onsite[:Q])

    # Continuous independent closed forms. Uniform rho_00=1 has self
    # integral8*pi*R^5/15. Compact g has exact polynomial rational moments.
    analytic=[]
    for order in (128,256):
        nodes,weights=np.polynomial.legendre.leggauss(order)
        ra,wa=R*(nodes+1)/2,R*weights/2
        tables=atomic_radial_metrics(ra,wa,[0,1,2],support_radius=R,fft_points=1,cell_volume=1)
        uniform=float(np.ones(order)@tables['delta_metric'][0]@np.ones(order))/2
        uniform_error=abs(uniform/(8*np.pi*R**5/15)-1)
        errors=[]
        for l in range(3):
            coefficients_poly=[Fraction((-1)**a*comb(6,a)) for a in range(7)]
            normalization=sum(c/Fraction(2*l+3+2*a) for a,c in enumerate(coefficients_poly))
            double=sum(c*d/Fraction((2*l+3+2*b)*(2*l+5+2*a+2*b))
                       for a,c in enumerate(coefficients_poly) for b,d in enumerate(coefficients_poly))
            exact=4*np.pi/(2*l+1)*2*R**(-2*l-1)*float(double/(normalization**2))
            g=tables['compensation_shapes'][l]
            measured=float(g@tables['delta_metric'][l]@g)/2
            errors.append(abs(measured/exact-1))
        analytic.append(dict(order=order,uniform_relative_error=uniform_error,polynomial_compensation_relative_errors=errors))
    assert analytic[1]['uniform_relative_error'] < analytic[0]['uniform_relative_error']/3
    assert max(analytic[1]['polynomial_compensation_relative_errors']) < max(analytic[0]['polynomial_compensation_relative_errors'])/3
    assert analytic[1]['uniform_relative_error'] < 5e-5
    assert max(analytic[1]['polynomial_compensation_relative_errors']) < 2e-4
    return dict(q_logical=Q,q_carrier=Qp,mu=mu,G_logical=ng,G_carrier=8,
        radial_points=nr,atoms=na,LM=nh,max_fourier_error=fourier_error,
        max_compensation_fourier_error=compensation_error,max_padded_fourier_abs=padded_max,
        max_dense_onsite_error=onsite_error,onsite_hermiticity=hermitian,
        poisoned_q_padding_does_not_move_physical_onsite=True,invalid_LM_sets_refused=True,
        analytic_radial_convergence=analytic)


def check_projection_and_samples(runtime):
    import numpy as np
    import jax
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import device_put_process_local,gather_to_host
    from common.bispinor_init import lift_to_4spinor,NORMALIZED_RKB_LIFT
    from psp.augmented_samples import atomic_projection_table,make_atomic_projection,make_sample_correction
    from psp.augmentation_spinors import atomic_pauli_fourier,spinor_function_labels

    mesh=runtime.mesh
    nodes,weights=np.polynomial.legendre.leggauss(64)
    r,w=2*(nodes+1),2*weights
    ell=np.array([0,0,1,1,1]);kappa=np.array([-1,-1,1,1,-2])
    ps=np.column_stack((r*np.exp(-r*r),r*(1-.4*r*r)*np.exp(-.6*r*r)*(1+.3j),
        r*r*np.exp(-.8*r*r)*(1-.2j),r*r*(1-.2*r*r)*np.exp(-.5*r*r)*(1+.7j),r*r*np.exp(-.9*r*r)))
    data=dict(r=r,weights_dr=w,l=ell,kappa=kappa,ps_u=ps)
    dual=np.zeros_like(ps)
    dual_error=0.
    for k in np.unique(kappa):
        columns=np.flatnonzero(kappa==k);waves=ps[:,columns]
        S=waves.conj().T@(w[:,None]*waves)
        dual[:,columns]=waves@np.linalg.inv(S)
        dual_error=max(dual_error,float(np.max(np.abs(dual[:,columns].conj().T@(w[:,None]*waves)-np.eye(len(columns))))))
    assert dual_error < 2e-13
    rng=np.random.default_rng(737130)
    nparent,nb,ng,mu=2,6,16,10
    reciprocal=rng.normal(size=(ng,3))*.7
    k=np.array([[.23,-.19,.31],[-.11,.17,.29]])
    K=reciprocal[None]+k[:,None]
    center=np.array([.7,-.3,.2]);volume=120.
    table=np.stack([atomic_projection_table(data,v,center_cart=center,cell_volume=volume) for v in K])
    expected=np.stack([atomic_pauli_fourier(dual/r[:,None],r,w,ell,kappa,v,center_cart=center).conj()/np.sqrt(volume) for v in K])
    table_error=float(np.max(np.abs(table-expected)))
    assert table_error < 2e-13,table_error
    source=rng.normal(size=(nparent,nb,2,ng))+1j*rng.normal(size=(nparent,nb,2,ng))
    coefficients_ref=np.einsum('pnsg,pisg->pni',source,expected)
    spec=P(None,None,None,('x','y'))
    put=lambda a,spec=P():device_put_process_local(np.asarray(a),NamedSharding(mesh,spec))
    project=make_atomic_projection(mesh)
    coefficients=project(put(source,spec),put(table,spec))
    projection_error=float(np.max(np.abs(np.asarray(gather_to_host(coefficients))-coefficients_ref)))
    assert projection_error < 3e-13,projection_error
    lift=np.asarray(lift_to_4spinor(source,K,np.zeros((nparent,3)),np.eye(3),representation=NORMALIZED_RKB_LIFT))
    normalized_table=np.stack([atomic_projection_table(data,v,center_cart=center,cell_volume=volume,normalized_rkb_source=True) for v in K])
    normalized=project(put(lift[:,:,:2],spec),put(normalized_table,spec))
    normalized_error=float(np.max(np.abs(np.asarray(gather_to_host(normalized))-coefficients_ref)))
    assert normalized_error < 3e-13,normalized_error

    # Non-TRIM k, explicit image translations, both admitted face layouts.
    labels=spinor_function_labels(ell,kappa)
    smooth=rng.normal(size=(nparent,nb,4,mu))+1j*rng.normal(size=(nparent,nb,4,mu))
    delta=rng.normal(size=(len(labels),4,mu))*.1+1j*rng.normal(size=(len(labels),4,mu))*.1
    images=np.array([[0,0,0],[1,0,0],[-1,2,0],[0,0,1],[2,-1,0],[-2,1,1],[1,1,-1],[0,-1,0],[1,0,2],[-1,-1,-1]])
    phases=np.exp(2j*np.pi*(k@images.T))
    reference=smooth+np.einsum('pni,ism,pm->pnsm',coefficients_ref,delta,phases)
    wrong_phase=smooth+np.einsum('pni,ism,pm->pnsm',coefficients_ref,delta,phases.conj())
    phase_signal=float(np.linalg.norm(reference-wrong_phase)/np.linalg.norm(reference))
    assert phase_signal > .03,phase_signal
    layouts=[]
    for face_spec in (P(None,'x',None,'y'),P(None,None,None,'y')):
        face=NamedSharding(mesh,face_spec)
        correct=make_sample_correction(mesh,face_sharding=face)
        dsh=P(None,None,face_spec[3]);psh=P(None,face_spec[3])
        actual=correct(put(smooth,face_spec),coefficients,put(delta,dsh),put(phases,psh))
        error=float(np.max(np.abs(np.asarray(gather_to_host(actual))-reference)))
        assert error < 5e-13,error
        zero=correct(put(smooth,face_spec),coefficients,put(np.zeros_like(delta),dsh),put(phases,psh))
        assert np.array_equal(np.asarray(gather_to_host(zero)),smooth)
        assert actual.sharding == face
        executable=correct.lower(put(smooth,face_spec),coefficients,put(delta,dsh),put(phases,psh)).compile()
        memory=executable.memory_analysis()
        layouts.append(dict(spec=str(face_spec),max_explicit_error=error,zero_exact=True,
            compiled_argument_bytes=memory.argument_size_in_bytes,compiled_output_bytes=memory.output_size_in_bytes,compiled_temporary_bytes=memory.temp_size_in_bytes))
    return dict(parent=nparent,bands=nb,G=ng,mu=mu,functions=len(labels),complex_dual_overlap_error=dual_error,
        max_projection_table_error=table_error,max_band_projection_error=projection_error,
        raw_vs_normalized_projection_error=normalized_error,wrong_image_phase_relative_signal=phase_signal,face_layouts=layouts)


def check_images():
    import numpy as np
    from psp.augmented_samples import atomic_image_geometry
    center=np.zeros(3);point=np.array([[.49,.49,0.]])
    skew=np.array([[1.,0,0],[3.,.3,0],[0,0,1.]])
    relative,image=atomic_image_geometry(point,center,skew)
    candidates=np.array(list(product(range(-6,7),repeat=3)))
    all_relative=(point[0]-candidates)@skew
    best=int(np.argmin(np.sum(all_relative**2,axis=1)))
    assert np.array_equal(image[0],candidates[best]),(image,candidates[best])
    assert np.array_equal(image[0],[2,0,0])
    translations=np.array([[7,-4,2],[-3,5,-1]])
    moved,images=atomic_image_geometry(point+translations,center,skew)
    assert np.max(np.abs(moved-relative)) < 3e-15
    assert np.array_equal(images,image+translations)
    a,R=9.,2.
    fcc=a/2*np.array([[0,1,1],[1,0,1],[1,1,0]])
    minimum_image_distance=a/np.sqrt(2)
    assert 2*R < minimum_image_distance
    offsets=np.array(list(product(range(-3,4),repeat=3)))
    centers=offsets@fcc
    nonzero=np.any(offsets!=0,axis=1)
    assert np.min(np.linalg.norm(centers[nonzero],axis=1)) == minimum_image_distance
    try:
        atomic_image_geometry(point,center,np.zeros((3,3)))
    except ValueError:
        pass
    else:
        raise AssertionError('singular lattice accepted by nearest-image helper')
    return dict(skew_counterexample_exact_image=image[0].tolist(),distance=float(np.linalg.norm(relative)),
        wrapped_translation_covariant=True,ordinary_fcc_minimum_image_distance=minimum_image_distance,
        compact_support_radius=R,unique_image_margin=minimum_image_distance-2*R,singular_lattice_refused=True)


if __name__=='__main__':
    from runtime import initialize_communicator_stack,run_main_and_finalize
    runtime=initialize_communicator_stack()
    def main():
        import jax
        assert int(runtime.mesh.size)==4
        result=dict(P=4,radial=check_radial(runtime),projection_and_samples=check_projection_and_samples(runtime),geometry=check_images(),
            scope='Synthetic seam/units/distribution evidence; no sidecar authentication, absolute Sigma_X convergence or recovered-core physics claim.')
        if jax.process_index()==0:
            print(json.dumps(result),flush=True)
            out=os.environ.get('ATOMIC_AUDIT_REPORT')
            if out:Path(out).write_text(json.dumps(result,indent=2)+'\n')
        return 0
    run_main_and_finalize(main)

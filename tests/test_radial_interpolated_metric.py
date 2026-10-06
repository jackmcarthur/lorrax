"""Analytic, PSD and representation-consistency audit of radial field metrics.

The CLI runs on a real P4 runtime, while these are deterministic host-side
species-precomputation tests. No atomic-data or absolute GW accuracy claim.
"""
from fractions import Fraction
import json
from math import comb
import os
from pathlib import Path
import time


def exact_compensation_energy(l, radius):
    """Closed polynomial double integral; does not call producer quadrature."""
    import numpy as np
    coefficients=[Fraction((-1)**a*comb(6,a)) for a in range(7)]
    norm=sum(c/Fraction(2*l+3+2*a) for a,c in enumerate(coefficients))
    inner=sum(c*d/Fraction((2*l+3+2*b)*(2*l+5+2*a+2*b))
              for a,c in enumerate(coefficients) for b,d in enumerate(coefficients))
    return 8*np.pi/(2*l+1)*radius**(-2*l-1)*float(inner/norm**2)


def map_for_degree(tables, degree_row):
    import numpy as np
    result=np.array(tables['interpolation_map'],copy=True)
    result[:tables['origin_row_count'],0]=tables['origin_factors'][degree_row]
    return result


def check_closed_forms():
    import numpy as np
    from isdf.augmentation import radial_coulomb_metric_interpolated
    radius=1.7
    # r^l is exact both in the origin panel and a degree-five interpolant.
    grid=radius*np.array([.007,.037,.11,.22,.39,.61,.82,1.])
    tables=radial_coulomb_metric_interpolated(grid,[4,0,2,1,3,0],support_radius=radius,
                                             interpolation_degree=5)
    energy_errors=[];moment_errors=[];quadrature_errors=[]
    for row,l in enumerate(tables['degrees']):
        rho=grid**l
        exact_energy=8*np.pi*radius**(2*l+5)/((2*l+1)*(2*l+3)*(2*l+5))
        measured=rho@tables['metric'][row]@rho
        energy_errors.append(float(abs(measured/exact_energy-1)))
        moment=tables['moments'][row]@rho
        exact_moment=radius**(2*l+3)/(2*l+3)
        moment_errors.append(float(abs(moment/exact_moment-1)))
        sample_map=map_for_degree(tables,row)
        integrated=(tables['quadrature_weights_dr']*tables['quadrature_radius']**(l+2))@sample_map
        quadrature_errors.append(float(np.max(np.abs(integrated-tables['moments'][row]))))
    assert max(energy_errors)<3e-12,energy_errors
    assert max(moment_errors)<3e-12,moment_errors
    assert max(quadrature_errors)<3e-11,quadrature_errors
    assert np.all(tables['quadrature_weights_dr']>0)
    assert np.all(tables['quadrature_radius']>0)
    assert np.array_equal(tables['degrees'],np.arange(5))
    return dict(exact_multipoles=list(range(5)),max_relative_energy_error=max(energy_errors),
                max_relative_moment_error=max(moment_errors),
                max_interpolant_quadrature_moment_error=max(quadrature_errors))


def check_polynomial_convergence():
    import numpy as np
    from scipy.integrate import quad
    from scipy.special import spherical_jn
    from isdf.augmentation import radial_coulomb_metric_interpolated
    from isdf.atomic_coulomb import atomic_radial_metrics
    radius=2.3
    convergence=[]
    for n in (8,16,32,64):
        # A tiny first radius isolates interior interpolation convergence;
        # the physical origin extension remains a separately stated choice.
        grid=np.concatenate(([radius*1e-6],np.linspace(radius/(n-1),radius,n-1)))
        started=time.monotonic()
        tables=radial_coulomb_metric_interpolated(grid,[0,1,2],support_radius=radius)
        metric_seconds=time.monotonic()-started
        energy_errors=[];moment_errors=[];fourier_errors=[];symmetry_errors=[];min_eigenvalue_ratios=[]
        for row,l in enumerate(tables['degrees']):
            rho=grid**l*(1-(grid/radius)**2)**6
            moment=tables['moments'][row]@rho
            exact_moment=radius**(2*l+3)*sum((-1)**a*comb(6,a)/(2*l+3+2*a) for a in range(7))
            measured=rho@tables['metric'][row]@rho/moment**2
            energy_errors.append(float(abs(measured/exact_compensation_energy(int(l),radius)-1)))
            moment_errors.append(float(abs(moment/exact_moment-1)))
            sample_map=map_for_degree(tables,row)
            density=sample_map@rho/moment
            q=tables['quadrature_radius'];w=tables['quadrature_weights_dr']
            exact_normalization=exact_moment
            for wave in (.7/radius,5/radius,12/radius):
                got=np.dot(w*q*q*density,spherical_jn(int(l),wave*q))
                expected=quad(lambda x:x**(l+2)*(1-(x/radius)**2)**6*spherical_jn(int(l),wave*x)/exact_normalization,
                              0,radius,epsabs=1e-13,epsrel=1e-13)[0]
                # Absolute radial FT relative to the natural R^-l scale.
                fourier_errors.append(float(abs(got-expected)*radius**l))
            K=tables['metric'][row]
            symmetry_errors.append(float(np.max(np.abs(K-K.T))))
            eigen=np.linalg.eigvalsh(K)
            min_eigenvalue_ratios.append(float(eigen[0]/eigen[-1]))
        assert max(symmetry_errors)<1e-13
        assert min(min_eigenvalue_ratios)>-1e-13,min_eigenvalue_ratios
        convergence.append(dict(radial_samples=n,energy_relative_errors=energy_errors,
            moment_relative_errors=moment_errors,max_scaled_radial_fourier_error=max(fourier_errors),
            max_metric_asymmetry=max(symmetry_errors),minimum_eigenvalue_ratio=min(min_eigenvalue_ratios),
            metric_host_seconds=metric_seconds,quadrature_nodes=len(tables['quadrature_radius'])))
    assert max(convergence[-1]['energy_relative_errors'])<3e-5,convergence
    assert max(convergence[-1]['energy_relative_errors'])<max(convergence[-2]['energy_relative_errors'])/6,convergence
    assert convergence[-1]['max_scaled_radial_fourier_error']<convergence[-2]['max_scaled_radial_fourier_error']/6,convergence
    # Incumbent second-order shell metric with twice as many fit samples.
    n=128;x,w=np.polynomial.legendre.leggauss(n)
    r,wr=radius*(x+1)/2,radius*w/2
    incumbent=atomic_radial_metrics(r,wr,[0,1,2],support_radius=radius,fft_points=1,cell_volume=1)
    incumbent_error=[]
    for row,l in enumerate(incumbent['degrees']):
        g=incumbent['compensation_shapes'][row]
        measured=g@incumbent['delta_metric'][row]@g/2
        incumbent_error.append(float(abs(measured/exact_compensation_energy(int(l),radius)-1)))
    assert max(convergence[-1]['energy_relative_errors'])<max(incumbent_error)/10,(convergence,incumbent_error)
    return dict(interpolated=convergence,incumbent_shell_samples=n,
                incumbent_compensation_relative_errors=incumbent_error)


def check_psd_and_consistency():
    import numpy as np
    from isdf.augmentation import radial_coulomb_metric_interpolated
    radius=1.9
    grid=radius*np.array([.0003,.013,.042,.12,.27,.45,.63,.78,.9])
    ell=[0,1,4,7]
    rng=np.random.default_rng(71392)
    c=rng.normal(size=(4,len(grid)))+1j*rng.normal(size=(4,len(grid)))
    refinements=[]
    for order in (16,32,64):
        tables=radial_coulomb_metric_interpolated(grid,ell,support_radius=radius,
            interpolation_degree=3,quadrature_order=order)
        errors=[];mineig=[];hermiticity=[];bilinear=[]
        for row,l in enumerate(ell):
            K=tables['metric'][row]
            eig=np.linalg.eigvalsh(K)
            mineig.append(float(eig[0]/eig[-1]))
            block=c.conj()@K@c.T
            hermiticity.append(float(np.max(np.abs(block-block.T.conj()))))
            assert np.min(np.linalg.eigvalsh(block)) > -1e-10*np.max(np.diag(block).real)
            sample_map=map_for_degree(tables,row)
            integrated=(tables['quadrature_weights_dr']*tables['quadrature_radius']**(l+2))@sample_map
            errors.append(float(np.max(np.abs(integrated-tables['moments'][row]))))
            bilinear.append(block)
        assert min(mineig)>-1e-13
        assert max(hermiticity)<2e-12
        assert max(errors)<3e-10
        refinements.append(dict(order=order,max_moment_consistency_error=max(errors),
            minimum_eigenvalue_ratio=min(mineig),max_complex_bilinear_hermiticity=max(hermiticity),
            bilinear=bilinear))
    ref=refinements[-1].pop('bilinear')
    for entry in refinements[:-1]:
        matrices=entry.pop('bilinear')
        entry['max_relative_field_gram_refinement_error']=max(float(np.linalg.norm(a-b)/np.linalg.norm(b)) for a,b in zip(matrices,ref))
    assert refinements[0]['max_relative_field_gram_refinement_error']<1e-5,refinements
    assert refinements[1]['max_relative_field_gram_refinement_error']<1e-8,refinements
    return dict(nonuniform_grid=grid.tolist(),support_exceeds_last_sample=True,
                complex_density_count=len(c),gauss_refinements=refinements)


def check_independent_two_electron_integral():
    import numpy as np
    from scipy.integrate import quad
    from scipy.interpolate import BarycentricInterpolator
    from isdf.augmentation import radial_coulomb_metric_interpolated
    # Independent triangular two-electron integral. It reconstructs rho
    # with SciPy cardinal interpolation and integrates the min/max kernel,
    # without using the field-energy identity, moments or producer map.
    R=1.8
    r=R*np.array([.009,.031,.087,.19,.36,.57,.78,.94])
    rng=np.random.default_rng(332801)
    c=(rng.normal(size=len(r))+1j*rng.normal(size=len(r)))*.1
    tables=radial_coulomb_metric_interpolated(r,[0,2,5],support_radius=R)
    edges=np.concatenate(([0.],r,[R]))
    relative_errors=[]
    for row,l in enumerate(tables['degrees']):
        samples=c*(r/R)**l
        interpolants=[]
        for panel,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
            if panel==0:
                interpolants.append(lambda x,c0=samples[0],ell=l:c0*(x/r[0])**ell)
            else:
                left=np.searchsorted(r,(a+b)/2)-2
                left=max(0,min(left,len(r)-4))
                columns=np.arange(left,left+4)
                interpolants.append(BarycentricInterpolator(r[columns],samples[columns],rng=817))
        prefix=0j;triangular=0j
        for panel,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
            rho=interpolants[panel]
            def inner(x):
                return prefix+quad(lambda t:t**(l+2)*rho(t),a,x,epsabs=2e-13,epsrel=2e-13,complex_func=True)[0]
            triangular += quad(lambda x:np.conj(rho(x))*x**(1-l)*inner(x),a,b,
                               epsabs=2e-13,epsrel=2e-13,complex_func=True)[0]
            prefix += quad(lambda t:t**(l+2)*rho(t),a,b,epsabs=2e-13,epsrel=2e-13,complex_func=True)[0]
        exact=8*np.pi/(2*l+1)*triangular.real
        measured=np.vdot(samples,tables['metric'][row]@samples).real
        relative_errors.append(float(abs(measured/exact-1)))
    assert max(relative_errors)<2e-10,relative_errors
    return dict(degrees=tables['degrees'].tolist(),max_relative_energy_error=max(relative_errors),
                oracle='SciPy Barycentric physical density plus nested triangular min/max two-electron integral')


def check_invalid_inputs():
    import numpy as np
    from isdf.augmentation import radial_coulomb_metric_interpolated as metric
    cases=[([.1,.2,.2,.4],[0],.5,3,None),([0,.2,.3,.4],[0],.5,3,None),
           ([.1,.2,.3,.4],[.5],.5,3,None),([.1,.2,.3,.4],[0],.3,3,None),
           ([.1,.2,.3,.4],[0],.5,3.5,None),([.1,.2,.3,.4],[0],.5,3,1),
           ([.1,.2,.3,.4],[0],.5,3,2.5)]
    for r,l,R,d,q in cases:
        try:metric(r,l,support_radius=R,interpolation_degree=d,quadrature_order=q)
        except ValueError:pass
        else:raise AssertionError(f'invalid radial interpolant input accepted: {(r,l,R,d,q)}')
    return dict(refused_cases=len(cases))


if __name__=='__main__':
    from runtime import initialize_communicator_stack,run_main_and_finalize
    runtime=initialize_communicator_stack()
    def main():
        import jax
        assert int(runtime.mesh.size)==4
        result=dict(P=4,closed_forms=check_closed_forms(),polynomial_convergence=check_polynomial_convergence(),
            positive_metric=check_psd_and_consistency(),independent_two_electron=check_independent_two_electron_integral(),invalid_inputs=check_invalid_inputs(),
            scope='Host-side physical-density interpolation, analytic/convergence/PSD evidence on a P4 runtime; no sidecar fidelity, full fitting time or Sigma_X accuracy claim.')
        if jax.process_index()==0:
            print(json.dumps(result),flush=True)
            out=os.environ.get('RADIAL_INTERPOLATED_REPORT')
            if out:Path(out).write_text(json.dumps(result,indent=2)+'\n')
        return 0
    run_main_and_finalize(main)

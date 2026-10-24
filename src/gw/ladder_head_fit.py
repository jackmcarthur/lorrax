"""Certified scalar head for the tangential physical density ladder.

Exact Gamma actions feed the common positive Hermite/Ritz owner. Independent
held observations may be promoted only after failure; their intervals are
then bisected and measured again. No residue pin or pole repair is applied.
"""
import numpy as np
from common.log import get_logger

HEAD_FIT_POLICY = 'ladder-scalar-Hermite-held-v1'
log=get_logger(__name__)


def fit_ladder_head_actions(evaluate,recipe,legacy_z,*,gap,mesh=None):
    """Fit exact callback observations with bounded, measured refinement.

    ``evaluate(z)`` returns the scalar quadrature fields z/Wc/dWc_ds/vc0/M1/M3
    and optional per-point Xi tensors. The recipe owns body supports, eta,
    accuracy and the extra-site cap; no new deck control is introduced.
    """
    from common.units import RYD_TO_EV
    from .qsgw_head import head_quadrature_rule
    from .tangential_directions import _sample_point
    from .tangential_recipe import shared_real_pole_v1_r3b, shared_real_pole_gates_v1_r3b
    from .tangential_head_scalar import scalar_head_ritz,scalar_head_certificate

    if not np.isfinite(gap) or gap<=0:
        raise ValueError('GATE ladder_head_gap: a positive finite transition gap is required')
    eta=float(recipe['eta_ev'])/RYD_TO_EV
    scalar_tol=float(recipe['bank_rule_tolerance'])
    sample_tol=float(recipe['direction_cutoff'])
    static_tol=shared_real_pole_gates_v1_r3b['retained_subspace_moments']['threshold']
    cap=int(shared_real_pole_v1_r3b[recipe['accuracy']].get('sample_cap',
        shared_real_pole_v1_r3b['production']['sample_cap']))
    if not np.isfinite([eta,scalar_tol,sample_tol]).all() or min(eta,scalar_tol,sample_tol)<=0:
        raise ValueError('GATE ladder_head_budget: positive finite inherited accuracy budgets required')
    cache,fixed={},{}
    def agreed_max(value):
        if mesh is None:
            return np.asarray(value)
        from common.collectives import all_gather_processes
        return np.max(all_gather_processes(np.asarray(value)),axis=0)
    def observe(points):
        missing=list(dict.fromkeys(complex(z) for z in points if complex(z) not in cache))
        if not missing:
            return
        if any(not np.isfinite(z) or z.imag<0 or z.real<0 for z in missing):
            raise ValueError('GATE ladder_head_points: finite nonnegative upper-half-plane supports required')
        row=evaluate(np.asarray(missing,np.complex128))
        if not np.array_equal(np.asarray(row['z']),missing):
            raise ValueError('GATE ladder_head_points: exact callback coordinate identity required')
        for name in ('vc0','M1','M3'):
            if name in fixed and row[name]!=fixed[name]:
                raise ValueError(f'GATE ladder_head_operator: changed {name} across frequency batches')
            fixed[name]=row[name]
        if 'quadrature' in row:
            if 'quadrature' in fixed and row['quadrature']!=fixed['quadrature']:
                raise ValueError('GATE ladder_head_operator: changed quadrature receipt across batches')
            fixed['quadrature']=row['quadrature']
        for index,z in enumerate(missing):
            cache[z]={name:np.asarray(row[name])[index] for name in ('Wc','dWc_ds')}
            if 'Xi' in row:
                cache[z]['Xi']=np.asarray(row['Xi'])[index]
        if not all(np.isfinite(a).all() for entry in cache.values() for a in entry.values()):
            raise ValueError('GATE ladder_head_observation: finite exact actions required')
    def fields(points):
        points=tuple(dict.fromkeys(map(complex,points)))
        return dict(fixed,z=np.asarray(points),**{name:np.asarray([cache[z][name] for z in points])
                                                for name in ('Wc','dWc_ds')})
    def moment(order):
        xi,weights=head_quadrature_rule(order)
        points=tuple(map(complex,1j*xi));observe(points)
        return complex(np.sum(weights*(fields(points)['Wc']-cache[0j]['Wc'])/xi**2)/np.pi),points

    body_fit=[_sample_point(recipe,i) for i in recipe['fit_ids']]
    held=set(_sample_point(recipe,i) for i in recipe['held_ids'])
    # Gap and consumer broadening define the low-energy guard, rather than
    # this material's previously failed frequencies or another global grid.
    path=(0j,complex(gap,eta),complex(2*gap,eta))
    intervals=[(a,b) for a,b in zip(path[:-1],path[1:])]
    midpoints={.5*(a+b):(a,b) for a,b in intervals}
    held.update(midpoints)
    fit=set(map(complex,legacy_z))|set(body_fit)|set(path)
    # Body and legacy fitting roles belong to their own constructions. A
    # coincident head-held coordinate stays independent in THIS pencil.
    fit.difference_update(held)
    observe(sorted(fit|held,key=lambda z:(z.real,z.imag)))
    inverse24,quad24=moment(24)
    inverse48,quad48=moment(48)
    if not np.isfinite([inverse24,inverse48]).all():
        raise ValueError('GATE ladder_head_inverse_moment: finite imaginary quadrature required')
    inverse,reference_order=inverse48,48
    quad_error=float(agreed_max(abs(inverse48-inverse24)/max(abs(inverse48),np.finfo(float).tiny)))
    if quad_error>scalar_tol:
        inverse96,quad96=moment(96)
        quad_error=float(agreed_max(abs(inverse96-inverse48)/max(abs(inverse96),np.finfo(float).tiny)))
        inverse,reference_order=inverse96,96
        if quad_error>scalar_tol:
            raise ValueError('GATE ladder_head_inverse_moment: imaginary quadrature did not converge')
    fit.update(set(quad24)-held)
    attempts=[];promotions=[];extra=0
    for refinement in range(cap+1):
        chosen=None
        for cut in (1e-8,1e-10,1e-12):
            ordered=tuple(sorted(fit,key=lambda z:(z.real,z.imag)))
            f=fields(ordered)
            model=scalar_head_ritz(f['z'],f['Wc'],f['dWc_ds'],
                moment_m1=fixed['M1'],moment_m3=fixed['M3'],gram_keep=cut,mesh=mesh)
            observations=fields((*ordered,*sorted(held,key=lambda z:(z.real,z.imag))))
            certificate=scalar_head_certificate(model,observations,eta_ry=eta,
                static_value=cache[0j]['Wc'],inverse_m1=inverse,mesh=mesh)
            errors=agreed_max([certificate['sample_value_relative'],certificate['sample_slope_relative']])
            high=max(max(np.asarray(v)) for row in certificate['high_frequency'].values() for v in row.values())
            static_error,inverse_error,high=agreed_max([certificate['static_relative'],
                certificate['inverse_m1_relative'],high])
            moments_ok=(static_error<=static_tol and inverse_error<=scalar_tol and high<=scalar_tol)
            attempts.append(dict(refinement=refinement,gram_keep=cut,rank=len(model['Omega_p']),
                max_sample_error=float(np.max(errors)),static=float(static_error),
                inverse_m1=float(inverse_error),high_frequency=float(high)))
            log.info(f'ladder head: {len(ordered)} fit supports, {len(held)} held, '
                f'{len(model["Omega_p"])} poles; max value/slope error {np.max(errors):.2e}, '
                f'static {static_error:.2e}, inverse moment {inverse_error:.2e}')
            chosen=model,certificate,observations,errors,moments_ok
            if moments_ok and np.max(errors)<=sample_tol and held:
                model.update(sample_z=observations['z'],sample_Wc=observations['Wc'],
                    model='bse_resolvent_micro_shared_hermite_ritz',completion=True)
                model['provenance'].update(head_fit_policy=HEAD_FIT_POLICY,attempts=attempts,
                    fit_z=list(ordered),held_z=sorted(held,key=lambda z:(z.real,z.imag)),
                    moment_m1=inverse,moment_m1_unpinned=complex(np.sum(model['B_p']/model['Omega_p']**2)),
                    moment_rule=f'Gauss-Legendre order{reference_order}',moment_convergence=quad_error,
                    promotions=promotions,accuracy=dict(sample=sample_tol,static=static_tol,
                        inverse_m1=scalar_tol,high_frequency=scalar_tol))
                model['diagnostics'].update(certificate=certificate,
                    fit_condition=float(np.max(model['diagnostics']['reduction']['gram_condition'])),
                    fit_backward_error=float(np.max(model['diagnostics']['reduction']['retained_metric_relative'])),
                    fit_max_abs_residual=float(np.max(errors[0]*abs(observations['Wc']))))
                return model,dict(cache=cache,fields=fixed,attempts=attempts)
        model,certificate,observations,errors,moments_ok=chosen
        # First refine the inverse-moment support using already measured
        # reference nodes; no nonlinear residue adjustment is permitted.
        extra_quad=set(quad48 if reference_order==48 else quad96)-fit-held
        if not moments_ok and extra_quad:
            fit.update(extra_quad)
            continue
        bad=[complex(z) for z,e in zip(observations['z'],errors.T)
             if complex(z) in held and np.max(e)>sample_tol]
        if not bad:
            raise ValueError('GATE ladder_head_accuracy: sampled/static/inverse/HF certificate not met')
        new=[]
        for point in bad:
            if point not in midpoints:
                # Body held points acquire local neighbours from the exact
                # current support geometry; retain their held identity until
                # this failure, then refine the two adjacent segments.
                nearest=sorted(fit,key=lambda z:abs(z-point))[:2]
                pair=tuple(nearest)
            else:
                pair=midpoints.pop(point)
            held.remove(point);fit.add(point);promotions.append(point)
            for a,b in ((pair[0],point),(point,pair[1])):
                midpoint=.5*(a+b)
                if midpoint not in fit and midpoint not in held:
                    midpoints[midpoint]=(a,b);held.add(midpoint);new.append(midpoint)
        extra+=len(new)
        if extra>cap:
            raise ValueError('GATE ladder_head_refinement_cap: extra-site budget exhausted before certification')
        observe(new)
    raise ValueError('GATE ladder_head_refinement_cap: no certified scalar head')


def build_ladder_head_fit(restart,config,meta,mesh,head_resolver,*,identity):
    """Prepare one physical Gamma operator and certify its tangential head.

    The locally owned prepared payload and dipole survive adaptive batches;
    no large array is captured in a cache or reloaded for each refinement.
    """
    import hashlib
    import json
    from pathlib import Path
    from bse.w_ladder import prepare_ladder_head,LADDER_RESPONSE_CONVENTION
    from bse.head_resolvent import (build_head_rhs,head_prefactor,solve_head_tensor)
    from common.collectives import gather_to_host
    from vcoul import CoulombGeometry,get_kernel,scalar_resolvent_head
    from .head_correction import HeadSample,HeadResponseKind
    from .qsgw_head import IterationHeadSamples
    from .mpa.sample_plan import plan_z
    from .tangential_head import tangential_head_plan
    from .screening_bse import _GMRES_TOL_TIGHT,_GMRES_MAX_ITER

    if str(getattr(config.head.correction,'value',config.head.correction))!='full':
        raise ValueError('GATE ladder_head_policy: the scalar fit requires the full physical head')
    if any(getattr(config.head,name,None) is not None for name in ('vhead','whead_0freq','whead_imfreq')):
        raise ValueError('GATE ladder_head_override: physical response actions cannot fit constant overrides')
    wfn=head_resolver.wfn
    operator,data,dipole,delta=prepare_ladder_head(str(restart),mesh,
        input_file=config.input_file,head_dipole_path=str(Path(config.input_dir)/'dipole.h5'),
        head_n_occ=int(meta.nelec),include_w=True)
    rhs=build_head_rhs(dipole,operator.sh,physical_branches=True)
    nk=int(meta.nk_tot)
    pref=head_prefactor(float(meta.cell_volume),nk,int(wfn.nspin),int(meta.nspinor_wfnfile))
    grid=tuple(int(data[k]) for k in ('nkx','nky','nkz'))
    eps_c=np.asarray(gather_to_host(data['eps_c']))[:,:int(data['n_cond'])]
    eps_v=np.asarray(gather_to_host(data['eps_v']))[:,:int(data['n_val'])]
    gap=float(np.min(eps_c[:,:,None]-eps_v[:,None,:]))
    geometry=CoulombGeometry.from_wfn(wfn)
    kernel=get_kernel(config.sys_dim)
    tensors,solves={},[]
    symmetric=lambda a:.5*(np.asarray(a)+np.swapaxes(np.asarray(a),-1,-2))

    def evaluate(z):
        log.info(f'ladder head: solving {len(z)} Gamma frequencies with exact response actions')
        result=solve_head_tensor(operator,data,dipole,z,pref=pref,
            max_iter=_GMRES_MAX_ITER,tol=_GMRES_TOL_TIGHT,rhs=rhs,
            arm='micro_reducible_bse',delta_cvk=delta,
            with_slope=True,with_moments=not tensors)
        if (not np.isfinite(result.resids).all() or np.max(result.resids)>_GMRES_TOL_TIGHT
                or np.max(result.iters)>=_GMRES_MAX_ITER):
            raise ValueError('GATE ladder_head_solve: exact value/slope/reflection tolerance not met')
        xi,reflected=symmetric(result.xi),symmetric(result.xi_minus_z)
        norm=max(float(np.max(abs(xi))),np.finfo(float).tiny)
        even=float(np.max(abs(xi-reflected))/norm)
        if not np.isfinite(even) or even>2e-6:
            raise ValueError('GATE ladder_head_evenness: physical even-s response required')
        if not tensors:
            tensors.update({name:symmetric(a) for name,a in result.moments.items()})
            odd=[np.linalg.norm(tensors[a])/max(np.linalg.norm(tensors[b]),np.finfo(float).tiny)
                 for a,b in (('M0','M1'),('M2','M3'))]
            if not np.isfinite(odd).all() or max(odd)>2e-6:
                raise ValueError('GATE ladder_head_odd_moments: physical even-s response required')
        for point,tensor in zip(z,xi):
            if point.real==0:
                largest=float(np.linalg.eigvalsh(.5*(tensor+tensor.conj().T)).max())
                if not np.isfinite(largest) or largest>2e-6*norm:
                    raise ValueError('GATE ladder_head_sign: imaginary-axis response must be negative semidefinite')
        fields=scalar_resolvent_head(kernel,geometry,grid,
            dict(Xi=xi,dXi_ds=symmetric(result.d_xi_ds),moments=tensors),
            analytic_sphere=bool(getattr(config.head,'analytic_q0_sphere',config.head.head_minibz_average)))
        solves.append(dict(count=len(z),max_residual=float(np.max(result.resids)),
            max_iterations=int(np.max(result.iters)),evenness=even))
        return dict(fields,z=np.asarray(z),Xi=xi)

    plan=tangential_head_plan(config,meta.tangential_recipe,material_class='insulator')
    model,state=fit_ladder_head_actions(evaluate,meta.tangential_recipe,plan_z(plan),gap=gap,mesh=mesh)
    cache=state['cache'];bare=state['fields']['vc0']
    samples=tuple(HeadSample(vc0=complex(bare),wcoul0=complex(bare+row['Wc']),
        source='bse_resolvent_micro_shared_hermite',omega=z,S_cart=row['Xi'],
        response_kind=HeadResponseKind.MICRO_REDUCIBLE) for z,row in cache.items())
    head_resolver.install_samples(samples)
    point_hash=hashlib.sha256(np.asarray([[z.real,z.imag] for z in cache],dtype='<f8').tobytes()).hexdigest()
    head_identity=dict(identity,head_fit_policy=HEAD_FIT_POLICY,head_support_sha256=point_hash,
        head_gmres_tol=_GMRES_TOL_TIGHT,head_gmres_cap=_GMRES_MAX_ITER)
    # Keep all policy/support controls explicit even though the body restart
    # is reusable: every head is rebuilt/certified on the current operator.
    head_identity['head_accuracy_sha256']=hashlib.sha256(json.dumps(
        model['provenance']['accuracy'],sort_keys=True).encode()).hexdigest()
    model['provenance'].update(head_identity=head_identity,solves=solves,
        response_convention=LADDER_RESPONSE_CONVENTION,quadrature=state['fields']['quadrature'])
    return IterationHeadSamples(omegas=tuple(cache),samples=samples,
        sigma_energies_ry=np.empty((0,0)),sigma_occupations=np.empty((0,0)),
        efermi_ry=0.,scalar_model=model)

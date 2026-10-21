"""Bounded current-data refinement planning for a general W realization.

These plain helpers do NOT fit, repair or admit a model. The native builder
must evaluate the proposed new microscopic targets, retain its old samples,
and rerun every original fit/held/moment/growth check. A verified inverse at
one fitted pole proves COMPUTED-matrix nonsingularity ONLY at that point,
not physical-kernel invertibility without a propagated operator error bound,
nor global causal stability. No equilibrium/positive-residue premise is used.

STAGED, NOT SELECTED BY THE PRODUCTION BUILDER. Remaining native wiring:
extract its current-site planning/stream/Dyson loop into one reusable stage;
retain values across at most two fit attempts; compute an exact-point native
epsilon witness with fresh workspace at each candidate (the current explicit
diagnostic entrypoint is q0-only); call that SAME stage for ONLY the new
evaluate=True targets; add the independent L/R residue panels to the existing
direction tables; rebuild the current general pencil with invalidated held
frequency spaces when its dimensions grow; and rerun EVERY original/new
fit/held/moment/capacity/growth gate. Failure remains a normal refusal. The
functions below do not silently activate this unfinished native lifecycle.
``state_digest`` must bind the current E/U/rho, bare V, gamma, response
grid/window and source recipe, not just an iteration label or density hash.
"""
import numpy as np


MAX_REFINEMENT_ROUNDS = 2
MAX_REFINEMENT_POLES = 2


def residue_aligned_directions(left, right, pole_index, *, port_norm):
    """Return normalized current left/right residue directions ON DEVICE.

    Native full-port factors have shape ``[q,mu,K]``. The caller owns the
    normal whole-mesh JIT/sharding/admission and any frequency-panel padding.
    ``port_norm`` is its native ordered physical-port norm owner, retaining
    the mu axis (``[q,1,1]``); no unowned global reduction is introduced.
    These are two independent tangential directions, NOT a Hermitian residue
    replacement. All q rows remain present and no host mu array is formed.
    """
    import jax.numpy as jnp
    if left.shape!=right.shape or left.ndim!=3 or not 0<=pole_index<left.shape[-1]:
        raise ValueError('GATE shared_general_refinement_direction: compatible full-port factor column required')
    l,r=left[:,:,pole_index:pole_index+1],right[:,:,pole_index:pole_index+1]
    ln,rn=port_norm(l),port_norm(r)
    # Only exactly zero residue columns remain zero; no population/rank cut.
    return l/jnp.where(ln==0,1.,ln),r/jnp.where(rn==0,1.,rn)


def general_refinement_candidates(banks, minus_q, eta_ry, *, state_digest, round_index,
                                   support_height_ry=None):
    """Propose at most two observable, growth-refused canonical-q poles.

    ``banks[q]`` supplies small host ``poles_ry`` and ``residue_norm_bound``
    tables. Norms come from device contractions, never host port factors.
    The request is diagnostic only until an exact-point epsilon witness is
    authenticated. Exact-zero residues are omitted; no finite cutoff hides
    weak but nonzero poles. q/-q and self-q crossing duplicates are counted
    once, without assuming fixed-q even frequency symmetry.
    """
    support_height=eta_ry if support_height_ry is None else support_height_ry
    if (not isinstance(state_digest,str) or not state_digest or not np.isfinite(eta_ry)
            or eta_ry<=0 or not isinstance(round_index,int) or isinstance(round_index,bool)
            or not 0<=round_index<MAX_REFINEMENT_ROUNDS or not np.isfinite(support_height)
            or support_height<=0):
        raise ValueError('GATE shared_general_refinement: finite current identity/height and bounded round required')
    minus=np.asarray(minus_q)
    if (minus.ndim!=1 or minus.dtype.kind not in 'iu' or len(minus)!=len(banks)
            or np.any(minus<0) or np.any(minus>=len(minus))
            or not np.array_equal(minus[minus],np.arange(len(minus)))):
        raise ValueError('GATE shared_general_refinement: exact involutive q crossing map required')
    proposed=[]
    for q,bank in enumerate(banks):
        if q>minus[q]:
            continue
        p=np.asarray(bank['poles_ry'],complex)
        norm=np.asarray(bank['residue_norm_bound'],float)
        if (p.ndim!=1 or norm.shape!=p.shape or not np.isfinite(p).all()
                or not np.isfinite(norm).all() or np.any(norm<0)):
            raise ValueError('GATE shared_general_refinement: finite poles and residue norms required')
        seen=set()
        for i in np.flatnonzero((p.imag>=min(eta_ry,support_height)-1e-12)&(norm>0)):
            pole=complex(p[i])
            key=(abs(pole.real),pole.imag) if q==minus[q] else (pole.real,pole.imag)
            if key in seen:
                continue
            seen.add(key)
            proposed.append(dict(q=q,partner=int(minus[q]),pole_index=int(i),z_ry=pole,
                residue_norm_bound=float(norm[i]),state_digest=state_digest,round_index=round_index,
                required_height_ry=float(min(eta_ry,support_height)),
                status='NEEDS_EXACT_POINT_DIELECTRIC_WITNESS'))
    proposed.sort(key=lambda row:(-row['z_ry'].imag,-row['residue_norm_bound'],row['q'],row['pole_index']))
    return proposed[:MAX_REFINEMENT_POLES]


def authenticate_false_pole(candidate, witness, *, state_digest, gamma_ry):
    """Require the SAME current kernel and exact pole, not offset W agreement.

    ``witness`` is a scalar record from the existing native epsilon/inverse
    owner. A failed or unresolved inverse leaves the pole unclassified and
    must not trigger a supposedly 'spurious-pole' refinement.
    """
    try:
        z=witness['z_ry']; z=complex(*z) if isinstance(z,(list,tuple)) else complex(z)
        bound=witness['inverse_bounds']
        lower=float(bound['sigma_min_lower'])
        matches=(candidate['state_digest']==state_digest==witness['state_digest']
            and int(witness['q'])==candidate['q'] and z==candidate['z_ry']
            and float(witness['gamma_ry'])==float(gamma_ry)
            and witness['at_requested_site'] is True
            and float(witness['additional_collision_eta_ry'])==0.)
        n=int(bound['dimension']);padding=float(bound['padding_defect'])
        residual=float(bound['residual_spectral_upper_bound'])
        resolved=(bound['resolved'] is True and n>0 and np.isfinite(padding)
            and 0<=padding<=64*n*np.finfo(float).eps and np.isfinite(residual) and 0<=residual<1
            and np.isfinite(lower) and lower>0)
    except (KeyError,TypeError,ValueError,OverflowError) as error:
        raise ValueError('GATE shared_general_refinement_witness: complete exact-point native inverse witness required') from error
    if not matches or not resolved:
        raise ValueError('GATE shared_general_refinement_witness: current exact-point nonsingularity NOT established')
    return dict(q=candidate['q'],z_ry=z,state_digest=state_digest,gamma_ry=float(gamma_ry),
        sigma_min_lower=lower,classification='COMPUTED_DIELECTRIC_NONSINGULAR_AT_FITTED_POINT_ONLY',
        physical_kernel_pole_excluded=False,global_stability_proved=False)


def general_refinement_targets(candidates, witnesses, existing_sites, *, state_digest, gamma_ry, eta_ry):
    """Plan current-pole training plus independent nearby holds and mirrors.

    Only the positive-height representative needs a new full-q microscopic
    evaluation. Its negative-frequency partner uses the exact q/-q crossing
    gather/conjugation, NOT a fixed-q stationary reverse shortcut. All old
    sample IDs/roles are preserved. Collisions refuse rather than promote a
    held sample or invent independent evidence.
    """
    old=np.asarray(existing_sites,complex)
    if (old.ndim!=1 or not np.isfinite(old).all() or not np.isfinite(eta_ry)
            or eta_ry<=0 or len(candidates)>MAX_REFINEMENT_POLES or len(candidates)!=len(witnesses)):
        raise ValueError('GATE shared_general_refinement_targets: bounded finite current targets required')
    targets=[];proofs=[];used={complex(z):('original',None) for z in old}
    for candidate,witness in zip(candidates,witnesses):
        if not 0<=candidate['round_index']<MAX_REFINEMENT_ROUNDS:
            raise ValueError('GATE shared_general_refinement_targets: refinement round exhausted')
        proofs.append(authenticate_false_pole(candidate,witness,state_digest=state_digest,gamma_ry=gamma_ry))
        p=complex(candidate['z_ry']); delta=.25*min(float(eta_ry),p.imag)
        for role,z in (('fit',p),('held',p+delta*(1+1j))):
            if z in used:
                if used[z][0]==role:
                    continue  # One full-q value serves identical sites at different q.
                raise ValueError('GATE shared_general_refinement_targets: repeated target or fit/held collision; no role promotion')
            representative=len(targets)
            used[z]=(role,representative)
            targets.append(dict(z_ry=z,role=role,evaluate=True,candidate_q=candidate['q']))
            mirror=-z.conjugate()
            if mirror!=z:
                if mirror in used:
                    if used[mirror][0]==role:
                        continue
                    raise ValueError('GATE shared_general_refinement_targets: crossing target collision')
                used[mirror]=(role,len(targets))
                targets.append(dict(z_ry=mirror,role=role,evaluate=False,
                    crossing_from=representative,candidate_q=candidate['partner']))
    return dict(targets=targets,proofs=proofs,new_microscopic_target_count=sum(t['evaluate'] for t in targets),
        max_rounds=MAX_REFINEMENT_ROUNDS,max_poles_per_round=MAX_REFINEMENT_POLES,
        state_digest=state_digest,status='PLANNED_NOT_EXECUTED_NOT_ADMITTED',
        required_checks='all original and new full-W fit/held, moments, growth, and capacity')


def append_general_refinement_samples(state, plan, evaluated_values, *, crossing):
    """Append ONLY new runtime values, preserving old distributed operands.

    ``evaluated_values`` contains one current full-q native W array per
    evaluate=True target. ``crossing(value)`` is the native device q/-q
    gather/conjugation owner. It is deliberately injected rather than a
    NumPy conversion or host-array implementation. This helper neither calls
    chi nor admits a fit; execution/fresh-capacity ownership remains native.
    """
    if (state['state_digest']!=plan['state_digest']
            or not 0<=state.get('refinement_rounds',0)<MAX_REFINEMENT_ROUNDS
            or len(evaluated_values)!=plan['new_microscopic_target_count']):
        raise ValueError('GATE shared_general_refinement_append: current identity/round/value count mismatch')
    sites=list(map(complex,state['sites']));values=list(state['values'])
    fit=list(state['fit_positions']);held=list(state['held_positions'])
    if (len(sites)!=len(values) or set(fit)&set(held)
            or sorted(fit+held)!=list(range(len(sites))) or not values):
        raise ValueError('GATE shared_general_refinement_append: complete original sample roles required')
    shape=values[0].shape;dtype=values[0].dtype
    supplied=iter(evaluated_values);appended=[]
    for target in plan['targets']:
        value=next(supplied) if target['evaluate'] else crossing(appended[target['crossing_from']])
        if value.shape!=shape or value.dtype!=dtype or target['z_ry'] in sites:
            raise ValueError('GATE shared_general_refinement_append: current full-q shape/dtype or duplicate target')
        (fit if target['role']=='fit' else held).append(len(sites))
        sites.append(target['z_ry']);values.append(value);appended.append(value)
    return dict(state,sites=np.asarray(sites,complex),values=values,fit_positions=fit,held_positions=held,
        refinement_rounds=state.get('refinement_rounds',0)+1,
        status='CURRENT_SAMPLES_EXTENDED_FIT_AND_ADMISSION_STILL_REQUIRED')


def evaluate_general_refinement(state, plan, *, evaluate_current, crossing):
    """Invoke a native provider for ONLY new targets, then retain old values.

    The supplied provider is the existing current-site chi/Dyson stage after
    its normal scalar certification and compiled/provider admission, not a
    second response implementation. It returns values in the requested order
    and a receipt binding the current state and exact physical sites. This
    orchestration is testable now; production stage extraction is still TODO.
    """
    if (state['state_digest']!=plan['state_digest']
            or not 0<=state.get('refinement_rounds',0)<MAX_REFINEMENT_ROUNDS):
        raise ValueError('GATE shared_general_refinement_evaluate: stale identity or round exhausted BEFORE evaluation')
    sites=np.asarray([t['z_ry'] for t in plan['targets'] if t['evaluate']],complex)
    if not len(sites):
        raise ValueError('GATE shared_general_refinement_evaluate: no new microscopic targets')
    values,receipt=evaluate_current(sites)
    if (receipt.get('state_digest')!=state['state_digest']
            or not np.array_equal(np.asarray(receipt.get('sites_ry'),complex),sites)
            or receipt.get('native_capacity_admitted') is not True
            or receipt.get('scalar_rules_certified') is not True):
        raise ValueError('GATE shared_general_refinement_evaluate: exact current native target/certificate/admission receipt required')
    extended=append_general_refinement_samples(state,plan,values,crossing=crossing)
    extended['incremental_evaluation_receipt']=receipt
    return extended

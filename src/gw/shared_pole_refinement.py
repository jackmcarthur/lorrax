"""Bounded current-data refinement planning for a general W realization.

These plain helpers do NOT fit, repair or admit a model. The native builder
must evaluate the proposed new microscopic targets, retain its old samples,
and rerun every original fit/held/growth gate and moment measurement. Here W
means the correlation Wc returned by the native Dyson/general-store owner,
not bare V plus Wc. A verified inverse at
one fitted pole proves COMPUTED-matrix nonsingularity ONLY at that point,
not physical-kernel invertibility without a propagated operator error bound,
nor global causal stability. No equilibrium/positive-residue premise is used.

These helpers do not select or admit a production path. Native orchestration
must retain values across at most two refinement rounds; compute an exact-point
epsilon witness with current admitted workspace at each candidate; call the
SAME current-site sampling stage for ONLY the new
evaluate=True targets; add the independent L/R residue panels to the existing
direction tables; rebuild the current general pencil with invalidated held
frequency spaces when its dimensions grow; and rerun EVERY original/new
fit/held/capacity/growth gate and the existing moment measurements (which
are NOT a full-port moment admission certificate). Failure remains a normal refusal. The
functions below do not silently activate this unfinished native lifecycle.
``state_digest`` must bind the current E/U/rho, bare V, gamma, response
grid/window and source recipe, not just an iteration label or density hash.
"""
import numpy as np


MAX_REFINEMENT_ROUNDS = 2
MAX_REFINEMENT_POLES = 2


def _support_signature(sites, supports):
    """Validate explicit value-only support sides, including old mirrors.

    Original mirror actions may reference the original positive-frequency
    sample with ``sample_crossing=True``. New mirror values are explicit
    appended samples. In both cases ``z_ry`` names the ACTUAL tangent site.
    Runtime port directions/actions are caller-owned, never stored here.
    """
    if supports is None:
        raise ValueError('GATE shared_general_refinement_supports: explicit original support records required')
    signature=[];seen=set();left=[];right=[]
    for row in supports:
        try:
            i=row['sample_index'];side=row['side'];panel=row['panel_id']
            if isinstance(i,bool) or not isinstance(i,(int,np.integer)) or not 0<=i<len(sites):
                raise ValueError
            if side not in ('L','R') or not isinstance(panel,str) or not panel or panel in seen:
                raise ValueError
            crossing=row.get('sample_crossing',False)
            if not isinstance(crossing,(bool,np.bool_)):
                raise ValueError
            expected=-complex(sites[i]).conjugate() if crossing else complex(sites[i])
            z=complex(row.get('z_ry',expected))
            if not np.isfinite(z) or z!=expected:
                raise ValueError
        except (KeyError,TypeError,ValueError,OverflowError) as error:
            raise ValueError('GATE shared_general_refinement_supports: stable sample/side/panel and exact site required') from error
        descriptor=tuple((key,row.get(key)) for key in ('candidate_index','candidate_q',
            'pole_index','q','residue_factor','mirror_from','direction_crossing'))
        seen.add(panel);signature.append((int(i),side,panel,z,bool(crossing),descriptor))
        (left if side=='L' else right).append(z)
    for mu in left:
        for lam in right:
            # Same floating-point confluence scale as the finite owner. No
            # derivative data exist for this value-only general pencil.
            if abs(mu-lam)<=8*np.finfo(float).eps*max(1.,abs(mu),abs(lam)):
                raise ValueError('GATE shared_general_refinement_supports: L/R support collision or unresolved confluence')
    return tuple(signature)


def _independent_refinement_holds(sites,supports,held_indices):
    """A held value must be distinct from EVERY training tangent site."""
    signature=_support_signature(sites,supports)
    for index in held_indices:
        z=complex(sites[index])
        if any(abs(z-s[3])<=8*np.finfo(float).eps*max(1.,abs(z),abs(s[3])) for s in signature):
            raise ValueError('GATE shared_general_refinement_targets: independent held/training collision')


def _direction_signature(supports):
    """Bind declared NEW residue directions to their exact candidate/panel."""
    keys=('sample_index','z_ry','side','panel_id','candidate_index','candidate_q',
          'pole_index','q','residue_factor','sample_crossing','mirror_from','direction_crossing')
    try:
        return tuple(tuple(row[key] for key in keys) for row in supports)
    except (KeyError,TypeError) as error:
        raise ValueError('GATE shared_general_refinement_supports: complete current direction descriptors required') from error


def _plan_geometry_signature(targets,candidate_sites):
    """Bind sample roles/holds and the candidate-to-raw-Pi reuse join."""
    keys=('candidate_index','candidate_q','pole_index','partner_q','z_ry',
          'right','right_mirror','left','left_mirror','held','held_mirror')
    try:
        return (tuple((t['sample_index'],t['z_ry'],t['role'],t['evaluate'],t.get('crossing_from'))
                      for t in targets),
                tuple(tuple(row[key] for key in keys) for row in candidate_sites))
    except (KeyError,TypeError) as error:
        raise ValueError('GATE shared_general_refinement_targets: complete sample/candidate geometry required') from error


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


def general_refinement_preflight(candidates, existing_sites, *, state_digest,
                                 eta_ry, existing_supports):
    """Validate ONLY geometry BEFORE any expensive native epsilon probe.

    RIGHT training is at p, LEFT at p+delta*(1+i), and an independent held
    point at p+delta*(-1+2i), delta=.25*min(eta,Im(p)). SAME-side q/-q
    crossing supplies mirror supports, not a transpose or even-frequency
    premise. Coincident pure-imaginary samples are evaluated once while both
    direction descriptors remain. At most six new full-q sites per round.
    Original support records are required because their mirror sites need
    not be separate entries in the original physical sample table. No
    numerical witness is fabricated; this plan CANNOT evaluate/append data.
    """
    old=np.asarray(existing_sites,complex)
    if (old.ndim!=1 or not np.isfinite(old).all() or not np.isfinite(eta_ry)
            or len(set(map(complex,old)))!=len(old) or eta_ry<=0
            or not isinstance(state_digest,str) or not state_digest
            or not 0<len(candidates)<=MAX_REFINEMENT_POLES
            or any(c['state_digest']!=state_digest for c in candidates)):
        raise ValueError('GATE shared_general_refinement_targets: bounded finite current targets required')
    original_signature=_support_signature(old,existing_supports)
    targets=[];supports=[];proofs=[];candidate_sites=[]
    original_sites=set(map(complex,old))|{r[3] for r in original_signature}
    used={};rounds={c['round_index'] for c in candidates}
    if len(rounds)!=1:
        raise ValueError('GATE shared_general_refinement_targets: candidates must belong to ONE current round')
    def sample(z,role,*,mirror_of=None):
        if z in original_sites:
            raise ValueError('GATE shared_general_refinement_targets: original sample/support collision; no role promotion')
        if z in used:
            index=used[z]
            if targets[index]['role']!=role:
                raise ValueError('GATE shared_general_refinement_targets: fit/held collision; no role promotion')
            return index
        index=len(targets);used[z]=index
        row=dict(z_ry=z,role=role,sample_index=len(old)+index,evaluate=mirror_of is None)
        if mirror_of is not None:
            row['crossing_from']=mirror_of
        targets.append(row)
        return index
    for ci,candidate in enumerate(candidates):
        if not 0<=candidate['round_index']<MAX_REFINEMENT_ROUNDS:
            raise ValueError('GATE shared_general_refinement_targets: refinement round exhausted')
        p=complex(candidate['z_ry']); delta=.25*min(float(eta_ry),p.imag)
        if not np.isfinite(p) or p.imag<=0 or not np.isfinite(delta) or delta<=0:
            raise ValueError('GATE shared_general_refinement_targets: positive-height candidate required')
        mapping=dict(candidate_index=ci,candidate_q=int(candidate['q']),
            pole_index=int(candidate['pole_index']),partner_q=int(candidate['partner']),z_ry=p)
        for side,z in (('R',p),('L',p+delta*(1+1j)),(None,p+delta*(-1+2j))):
            role='held' if side is None else 'fit'
            representative=sample(z,role)
            mirror=-z.conjugate()
            reflected=sample(mirror,role,mirror_of=representative)
            name={'R':'right','L':'left',None:'held'}[side]
            mapping[name]=len(old)+representative
            mapping[name+'_mirror']=len(old)+reflected
            if side is not None:
                panel=f"refine:{candidate['round_index']}:{ci}:{side}"
                common=dict(side=side,candidate_index=ci,candidate_q=int(candidate['q']),
                    pole_index=int(candidate['pole_index']),
                    residue_factor='right' if side=='R' else 'left',sample_crossing=False)
                supports.append(dict(common,sample_index=len(old)+representative,z_ry=z,
                    panel_id=panel,q=int(candidate['q']),mirror_from=None,direction_crossing=False))
                # Even at pure imaginary/self-q sites the complex conjugate
                # direction can be independent: dedup SAMPLE, not PANEL.
                supports.append(dict(common,sample_index=len(old)+reflected,z_ry=mirror,
                    panel_id=panel+':mirror',q=int(candidate['partner']),
                    mirror_from=panel,direction_crossing=True))
        candidate_sites.append(mapping)
    all_sites=np.r_[old,np.asarray([t['z_ry'] for t in targets])]
    _independent_refinement_holds(all_sites,list(existing_supports)+supports,
        [t['sample_index'] for t in targets if t['role']=='held'])
    count=sum(t['evaluate'] for t in targets)
    if count>3*MAX_REFINEMENT_POLES:
        raise ValueError('GATE shared_general_refinement_targets: new-site budget exceeded')
    return dict(targets=targets,supports=supports,proofs=proofs,candidate_sites=candidate_sites,
        support_descriptor_signature=_direction_signature(supports),new_microscopic_target_count=count,
        target_geometry_signature=_plan_geometry_signature(targets,candidate_sites),
        original_sites_ry=old.copy(),original_support_signature=original_signature,
        round_index=next(iter(rounds)),support_policy='distinct-LR-same-side-crossing-v1',
        max_rounds=MAX_REFINEMENT_ROUNDS,max_poles_per_round=MAX_REFINEMENT_POLES,
        max_new_microscopic_targets=3*MAX_REFINEMENT_POLES,
        state_digest=state_digest,status='GEOMETRY_ONLY_NOT_AUTHENTICATED',authentication_required=True,
        required_checks='all original and new full-Wc fit/held, growth, and capacity gates',
        required_measurements='existing full-port moment defects; not a full-moment admission certificate')


def general_refinement_targets(candidates, witnesses, existing_sites, *, state_digest,
                               gamma_ry, eta_ry, existing_supports):
    """Authenticate exact-point witnesses on the SAME preflight geometry.

    Call ``general_refinement_preflight`` BEFORE querying raw Pi(p). This
    repeats only the cheap deterministic geometry owner after real witnesses
    exist. There is no bypass or synthetic witness for the expensive stage.
    """
    plan=general_refinement_preflight(candidates,existing_sites,state_digest=state_digest,
        eta_ry=eta_ry,existing_supports=existing_supports)
    if len(witnesses)!=len(candidates):
        raise ValueError('GATE shared_general_refinement_targets: one real witness per candidate required')
    proofs=[authenticate_false_pole(c,w,state_digest=state_digest,gamma_ry=gamma_ry)
            for c,w in zip(candidates,witnesses)]
    return dict(plan,proofs=proofs,authentication_required=False,status='PLANNED_NOT_EXECUTED_NOT_ADMITTED')


def _validate_refinement_state(state,plan):
    """Fail before a provider on changed roles, graph, or target indices."""
    sites=list(map(complex,state['sites']));values=state['values']
    if (plan.get('authentication_required') is not False
            or plan.get('status')!='PLANNED_NOT_EXECUTED_NOT_ADMITTED'
            or len(plan.get('proofs',()))!=len(plan['candidate_sites'])):
        raise ValueError('GATE shared_general_refinement_witness: exact-point authentication required BEFORE evaluation')
    if _plan_geometry_signature(plan['targets'],plan['candidate_sites'])!=plan['target_geometry_signature']:
        raise ValueError('GATE shared_general_refinement_targets: changed candidate/sample geometry BEFORE evaluation')
    fit=list(state['fit_positions']);held=list(state['held_positions'])
    supports=list(state.get('supports',()))
    if (len(sites)!=len(values) or set(fit)&set(held)
            or sorted(fit+held)!=list(range(len(sites))) or not values
            or any(s['sample_index'] not in fit for s in supports)):
        raise ValueError('GATE shared_general_refinement_append: complete original sample/support roles required BEFORE evaluation')
    if (not np.array_equal(sites,plan['original_sites_ry'])
            or _support_signature(sites,supports)!=plan['original_support_signature']):
        raise ValueError('GATE shared_general_refinement_append: changed original samples/support roles BEFORE evaluation')
    all_sites=sites.copy();new_roles={}
    for i,t in enumerate(plan['targets']):
        z=complex(t['z_ry'])
        if (t['sample_index']!=len(sites)+i or t['role'] not in ('fit','held')
                or not isinstance(t['evaluate'],(bool,np.bool_)) or not np.isfinite(z)
                or z.imag<=0 or z in all_sites):
            raise ValueError('GATE shared_general_refinement_targets: invalid target/role/index BEFORE evaluation')
        if not t['evaluate']:
            parent=t.get('crossing_from')
            if (isinstance(parent,bool) or not isinstance(parent,(int,np.integer))
                    or not 0<=parent<i or z!=-complex(plan['targets'][parent]['z_ry']).conjugate()
                    or t['role']!=plan['targets'][parent]['role']):
                raise ValueError('GATE shared_general_refinement_targets: invalid crossing association BEFORE evaluation')
        all_sites.append(z);new_roles[t['sample_index']]=t['role']
    if (sum(t['evaluate'] for t in plan['targets'])!=plan['new_microscopic_target_count']
            or not 0<plan['new_microscopic_target_count']<=3*MAX_REFINEMENT_POLES
            or any(new_roles.get(s['sample_index'])!='fit' for s in plan['supports'])):
        raise ValueError('GATE shared_general_refinement_targets: target count/support role mismatch BEFORE evaluation')
    if any(s['residue_factor']!=('right' if s['side']=='R' else 'left')
           or s.get('sample_crossing') is not False for s in plan['supports']):
        raise ValueError('GATE shared_general_refinement_supports: new side/factor association mismatch BEFORE evaluation')
    if _direction_signature(plan['supports'])!=plan['support_descriptor_signature']:
        raise ValueError('GATE shared_general_refinement_supports: changed current direction descriptor BEFORE evaluation')
    _independent_refinement_holds(all_sites,supports+plan['supports'],
        [t['sample_index'] for t in plan['targets'] if t['role']=='held'])


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
            or state.get('refinement_rounds',0)!=plan['round_index']
            or len(evaluated_values)!=plan['new_microscopic_target_count']):
        raise ValueError('GATE shared_general_refinement_append: current identity/round/value count mismatch')
    _validate_refinement_state(state,plan)
    sites=list(map(complex,state['sites']));values=list(state['values'])
    fit=list(state['fit_positions']);held=list(state['held_positions'])
    original_supports=list(state.get('supports',()))
    shape=values[0].shape;dtype=values[0].dtype
    supplied=iter(evaluated_values);appended=[]
    for target in plan['targets']:
        if target['sample_index']!=len(sites):
            raise ValueError('GATE shared_general_refinement_append: unstable sample index')
        value=next(supplied) if target['evaluate'] else crossing(appended[target['crossing_from']])
        if value.shape!=shape or value.dtype!=dtype or target['z_ry'] in sites:
            raise ValueError('GATE shared_general_refinement_append: current full-q shape/dtype or duplicate target')
        (fit if target['role']=='fit' else held).append(len(sites))
        sites.append(target['z_ry']);values.append(value);appended.append(value)
    supports=original_supports+plan['supports']
    _support_signature(sites,supports)
    return dict(state,sites=np.asarray(sites,complex),values=values,fit_positions=fit,held_positions=held,
        supports=supports,
        refinement_rounds=state.get('refinement_rounds',0)+1,
        status='CURRENT_SAMPLES_EXTENDED_FIT_AND_ADMISSION_STILL_REQUIRED')


def evaluate_general_refinement(state, plan, *, evaluate_current, crossing):
    """Invoke a native provider for ONLY new targets, then retain old values.

    The supplied provider is the existing current-site chi/Dyson stage after
    its normal scalar certification and compiled/provider admission, not a
    second response implementation. It returns values in the requested order
    and a receipt binding the current state and exact physical sites.
    The native builder owns explicit opt-in selection and every admission gate.
    """
    if (state['state_digest']!=plan['state_digest']
            or not 0<=state.get('refinement_rounds',0)<MAX_REFINEMENT_ROUNDS
            or state.get('refinement_rounds',0)!=plan['round_index']
            or not np.array_equal(state['sites'],plan['original_sites_ry'])
            or _support_signature(state['sites'],state.get('supports',()))!=plan['original_support_signature']):
        raise ValueError('GATE shared_general_refinement_evaluate: stale identity or round exhausted BEFORE evaluation')
    _validate_refinement_state(state,plan)
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

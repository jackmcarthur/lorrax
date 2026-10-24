"""A scalar-port head adapter to the common Hermite/Ritz construction.

No poles are moved and no residues are refitted. The shared positive pencil
owns its Gram cut, metric correction, Ritz solve and zero-weight policy.
The output is the scalar pole representation read by the existing Sigma
head executor, Wc(z)=sum 2 Omega B/(z**2-Omega**2).
"""
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np


def _all_valid(value,mesh):
    """Refuse together before the following scalar payload collective."""
    value=bool(np.all(value))
    if mesh is None:
        return value
    from common.collectives import all_gather_processes
    return bool(np.all(all_gather_processes(np.asarray([value],bool))))


def _one_parent_eigh(a,*,cut=None):
    """The scalar port has one parent; use its shared matrix solve policy."""
    from distrib_la import eigh_local
    if a.ndim!=3 or a.shape[0]!=1:
        raise ValueError('the scalar-head eigensolver expects exactly one parent')
    values,vectors=eigh_local(a[0])
    return values[None],vectors[None]


def _execute(program,args,*,mesh,side,label):
    """Admit the actual replicated small program and its provider workspace."""
    if not jax.config.jax_enable_x64:
        raise ValueError('GATE scalar_head_x64: physical head actions require the existing x64 mode')
    if mesh is None:
        from lxkit import device_vendor
        if device_vendor(jax.local_devices()[0])!='cpu':
            raise ValueError('GATE scalar_head_capacity: the active mesh is required on a GPU')
        return program(*args),None
    from distrib_la import local_eigh_workspace
    from lxkit import memory
    from runtime.aot_memory import compiled_new_bytes
    from common.gpu_utils import device_budget_bytes
    executable=program.lower(*args).compile()
    # Prepared H, its RHS, the body state and these concrete scalar inputs
    # coexist with this execution. The allocator room already charges all
    # of them; only the executable's NEW buffers and provider scratch remain.
    jax.block_until_ready(args)
    device_budget_bytes()
    room=memory.room(bucketed=False)
    vendor=local_eigh_workspace(mesh,1,int(side),np.complex128)
    total=compiled_new_bytes(executable)+vendor
    price=memory.priced(total,f'scalar head {label}: new program buffers + provider beside live operands')
    memory.book(price,stage='head.scalar.'+label)
    memory.admit('scalar_head.'+label,price,room)
    return executable(*args),int(total)


def _shared_poles(model,mesh):
    """One admitted scalar pole payload on every process, even at a cut tie."""
    if mesh is None:
        return model
    from common.collectives import all_gather_processes
    counts=np.asarray(all_gather_processes(np.asarray([len(model['Omega_p'])],np.int64))).reshape(-1)
    width=int(counts.max())
    payload=np.zeros((2,width),np.complex128)
    payload[0]=1
    payload[0,:len(model['Omega_p'])]=model['Omega_p']
    payload[1,:len(model['B_p'])]=model['B_p']
    # Physics payloads take the owner's device collective, not its host
    # metadata shortcut. Every process contributes an equal padded shape.
    gathered=all_gather_processes(jax.device_put(payload,jax.local_devices()[0]))
    model['Omega_p']=np.asarray(gathered)[0,0,:int(counts[0])].copy()
    model['B_p']=np.asarray(gathered)[0,1,:int(counts[0])].copy()
    model['provenance'].update(n_valid=int(counts[0]),rank_local_counts=counts.tolist())
    return model


@lru_cache(maxsize=None)
def _scalar_ritz_program():
    """Share each pencil geometry across runtime retention-cut refinements."""
    from .tangential_pencil import assemble_tangential_pencil
    from .tangential_reduction import reduce_tangential_pencil
    from .tangential_gates import apply_tangential_zero_policy, retained_moment_identity
    from .tangential_local import _mm
    from .tangential_recipe import shared_real_pole_gates_v1_r3b as canonical_gates
    @jax.jit
    def execute(points, values, slopes, m1, m3, gram_keep):
        gates = dict(canonical_gates,normalized_gram_keep=dict(
            canonical_gates['normalized_gram_keep'],threshold=gram_keep))
        direction = jnp.ones((1,1,points.size),jnp.complex128)
        finite = [(points, direction, values[None,None], slopes[None,None])]
        qi = jnp.ones((1,1,1),jnp.complex128)
        pencil = assemble_tangential_pencil(finite,
            (qi,m1.reshape(1,1,1),m3.reshape(1,1,1)),matmul=_mm)
        active = jnp.ones((1,points.size+1),bool)
        model, diagnostics, coefficients = reduce_tangential_pencil(
            pencil,active,eigh=_one_parent_eigh,matmul=_mm,gates=gates)
        model,zero = apply_tangential_zero_policy(model,gates=gates)
        selector = jnp.eye(points.size+1,dtype=jnp.complex128)[None,:,-1:]
        retained = retained_moment_identity(pencil,coefficients,model,selector,matmul=_mm)
        return model,diagnostics,zero,retained
    return execute


def scalar_head_ritz(z, values, slopes, *, moment_m1, moment_m3, gram_keep=None,mesh=None):
    """Construct a physical scalar head from exact response actions.

    ``z``, ``values`` and ``slopes`` are matching vectors, in Ry and bare
    head units; slopes differentiate s=z**2. ``moment_m1/m3`` are the two
    HIGH-frequency moments Wc(z)=2 M1/z**2+2 M3/z**4+..., not Sigma's
    inverse-frequency moment. Complex supports include their conjugate-s
    partners explicitly so the scalar real-residue closure is preserved.
    Mandatory shared physical gates fire here; sample and inverse-moment
    accuracy must additionally be measured by the caller before publication.
    ``gram_keep`` may refine the common retention cut downwards; all Gram
    validity, retained-metric and zero-weight predicates remain unchanged.
    """
    z,values,slopes = (np.asarray(a,np.complex128).reshape(-1) for a in (z,values,slopes))
    if not len(z) or values.shape != z.shape or slopes.shape != z.shape:
        raise ValueError('GATE scalar_head_actions: matching nonempty z/value/slope vectors required')
    if not all(np.isfinite(a).all() for a in (z,values,slopes,moment_m1,moment_m3)):
        raise ValueError('GATE scalar_head_actions: all samples and moments must be finite')
    points,outputs,derivatives = [],[],[]
    for point,value,derivative in zip(z*z,values,slopes):
        if any(point == known for known in points):
            raise ValueError('GATE scalar_head_actions: distinct squared support coordinates required')
        points.append(point);outputs.append(value);derivatives.append(derivative)
        if point.imag != 0:
            points.append(point.conjugate());outputs.append(value.conjugate());derivatives.append(derivative.conjugate())
    from .tangential_recipe import shared_real_pole_gates_v1_r3b as gates
    canonical_keep = gates['normalized_gram_keep']['threshold']
    gram_keep = canonical_keep if gram_keep is None else float(gram_keep)
    if not 0 < gram_keep <= canonical_keep:
        raise ValueError('GATE scalar_head_keep: retention may only refine the common cut')
    program = _scalar_ritz_program()
    args=(
        jnp.asarray(points),jnp.asarray(outputs),jnp.asarray(derivatives),
        jnp.asarray(moment_m1),jnp.asarray(moment_m3),jnp.asarray(gram_keep,jnp.float64))
    result,priced=_execute(program,args,mesh=mesh,side=len(points)+1,label='Ritz')
    model,reduction,zero,retained = jax.device_get(result)
    for name in ('gram_diagonal_positive','gram_valid','retained_metric_positive','ritz_eigh_finite'):
        if not _all_valid(reduction[name],mesh):
            raise ValueError(f'GATE scalar_head_{name}: shared Hermite/Ritz predicate failed; no PSD repair')
    if not _all_valid(zero['zero_policy'],mesh):
        raise ValueError('GATE scalar_head_zero_ritz: shared positive-weight policy failed; no pole clipping')
    if not _all_valid(all(np.all(np.isfinite(a)) and
            np.all(a<=gates['retained_subspace_moments']['threshold'])
            for a in retained.values()),mesh):
        raise ValueError('GATE scalar_head_retained_moments: shared retained-space identity failed')
    factors,poles,active = model
    poles = poles[0,active[0]]
    factors = factors[0,0,active[0]]
    omega = np.sqrt(poles)
    residues = np.abs(factors)**2/(2*omega)
    return _shared_poles(dict(Omega_p=omega.astype(np.complex128),B_p=residues.astype(np.complex128),
        diagnostics=dict(reduction={k:np.asarray(v).tolist() for k,v in reduction.items()},
            zero={k:np.asarray(v).tolist() for k,v in zero.items()},
            retained={k:np.asarray(v).tolist() for k,v in retained.items()}),
        provenance=dict(solve_mode='shared-Hermite-Ritz-scalar-v1',n_valid=len(omega),
            support_count=len(z),pencil_width=len(points)+1,gram_keep=gram_keep,
            compiled_price_bytes_per_rank=priced)),mesh)


@lru_cache(maxsize=None)
def _scalar_check_program():
    from .tangential_local import _mm, _round_check_equations
    from .tangential_recipe import shared_real_pole_gates_v1_r3b as gates

    @jax.jit
    def execute(omega,residue,z,values,slopes,bare,eta,m1,m3):
        model = (jnp.sqrt(2*omega*residue)[None,None],
                 omega[None]**2,jnp.ones((1,omega.size),bool))
        return _round_check_equations(model,(),(1/jnp.sqrt(bare)).reshape(1,1,1),
            values[None,:,None,None],slopes[None,:,None,None],z,eta,
            m1.reshape(1,1,1),m3.reshape(1,1,1),jnp.ones((1,1,1),jnp.complex128),
            matmul=_mm,eigh=_one_parent_eigh,gates=gates,ordered=False,return_samples=True)
    return execute


def scalar_head_certificate(model,observations,*,eta_ry,static_value,inverse_m1,mesh=None):
    """Measure the common passivity/held checks and scalar on-shell moments.

    ``observations`` holds z/Wc/dWc_ds and the same vc0/M1/M3 quadrature
    fields. Its rows are independent held observations or explicitly labelled
    training observations. This function refuses structural/passivity errors;
    the caller must enforce its declared accuracy budgets on the returned
    sample/static/inverse-moment errors before publishing a model.
    """
    omega,residue=(np.asarray(model[key],np.complex128) for key in ('Omega_p','B_p'))
    if omega.ndim!=1 or not len(omega) or residue.shape!=omega.shape:
        raise ValueError('GATE scalar_head_certificate: nonempty matching pole/residue vectors required')
    z,values,slopes=(np.asarray(observations[key],np.complex128).reshape(-1)
                     for key in ('z','Wc','dWc_ds'))
    bare_coulomb,moment_m1,moment_m3=(observations[key] for key in ('vc0','M1','M3'))
    if not len(z) or values.shape!=z.shape or slopes.shape!=z.shape:
        raise ValueError('GATE scalar_head_certificate: matching nonempty observations required')
    if (not np.isfinite(bare_coulomb) or np.real(bare_coulomb)<=0 or np.imag(bare_coulomb)!=0
            or not np.isfinite(eta_ry) or eta_ry<=0):
        raise ValueError('GATE scalar_head_certificate: positive bare Coulomb and eta required')
    if (not all(np.isfinite(a).all() for a in (omega,residue,z,values,slopes,
            moment_m1,moment_m3,static_value,inverse_m1)) or
            np.any(omega.imag!=0) or np.any(omega.real<=0) or
            np.any(residue.imag!=0) or np.any(residue.real<0)):
        raise ValueError('GATE scalar_head_certificate: finite real positive pole measure required')
    args=(
        jnp.asarray(omega.real),jnp.asarray(residue.real),jnp.asarray(z),
        jnp.asarray(values),jnp.asarray(slopes),jnp.asarray(np.real(bare_coulomb)),
        jnp.asarray(eta_ry),jnp.asarray(moment_m1),jnp.asarray(moment_m3))
    result,priced=_execute(_scalar_check_program(),args,mesh=mesh,side=1,label='certificate')
    passive,errors,reciprocity,moments,predicted=jax.device_get(result)
    if not _all_valid(passive['passivity'],mesh):
        raise ValueError('GATE scalar_head_passivity: shared Coulomb-normalized bounds failed')
    if not _all_valid(np.isfinite(errors).all() and np.all(reciprocity['passed']),mesh):
        raise ValueError('GATE scalar_head_samples: shared finite/reciprocity check failed')
    relative=lambda value,reference: float(abs(value-reference)/max(abs(reference),np.finfo(float).tiny))
    return dict(compiled_price_bytes_per_rank=priced,
        predicted_values=np.asarray(predicted)[:,0,0,0],
        passivity={key:np.asarray(a).tolist() for key,a in passive.items()},
        sample_value_relative=np.asarray(errors)[0,0].tolist(),
        sample_slope_relative=np.asarray(errors)[0,1].tolist(),
        reciprocity={key:np.asarray(a).tolist() for key,a in reciprocity.items()},
        high_frequency={key:{name:np.asarray(a).tolist() for name,a in row.items()}
                        for key,row in moments.items()},
        static_relative=relative(-2*np.sum(residue/omega),static_value),
        inverse_m1_relative=relative(np.sum(residue/omega**2),inverse_m1))

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


@lru_cache(maxsize=None)
def _scalar_ritz_program(gram_keep):
    from distrib_la import eigh_local
    from .tangential_pencil import assemble_tangential_pencil
    from .tangential_reduction import reduce_tangential_pencil
    from .tangential_gates import apply_tangential_zero_policy, retained_moment_identity
    from .tangential_local import _mm
    from .tangential_recipe import shared_real_pole_gates_v1_r3b as canonical_gates
    gates = dict(canonical_gates,normalized_gram_keep=dict(
        canonical_gates['normalized_gram_keep'],threshold=float(gram_keep)))

    def eigh(a, *, cut=None):
        return tuple(eigh_local(a))

    @jax.jit
    def execute(points, values, slopes, m1, m3):
        direction = jnp.ones((1,1,points.size),jnp.complex128)
        finite = [(points, direction, values[None,None], slopes[None,None])]
        qi = jnp.ones((1,1,1),jnp.complex128)
        pencil = assemble_tangential_pencil(finite,
            (qi,m1.reshape(1,1,1),m3.reshape(1,1,1)),matmul=_mm)
        active = jnp.ones((1,points.size+1),bool)
        model, diagnostics, coefficients = reduce_tangential_pencil(
            pencil,active,eigh=eigh,matmul=_mm,gates=gates)
        model,zero = apply_tangential_zero_policy(model,gates=gates)
        selector = jnp.eye(points.size+1,dtype=jnp.complex128)[None,:,-1:]
        retained = retained_moment_identity(pencil,coefficients,model,selector,matmul=_mm)
        return model,diagnostics,zero,retained
    return execute


def scalar_head_ritz(z, values, slopes, *, moment_m1, moment_m3, gram_keep=None):
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
    program = _scalar_ritz_program(gram_keep)
    model,reduction,zero,retained = jax.device_get(program(
        jnp.asarray(points),jnp.asarray(outputs),jnp.asarray(derivatives),
        jnp.asarray(moment_m1),jnp.asarray(moment_m3)))
    for name in ('gram_diagonal_positive','gram_valid','retained_metric_positive','ritz_eigh_finite'):
        if not np.all(reduction[name]):
            raise ValueError(f'GATE scalar_head_{name}: shared Hermite/Ritz predicate failed; no PSD repair')
    if not np.all(zero['zero_policy']):
        raise ValueError('GATE scalar_head_zero_ritz: shared positive-weight policy failed; no pole clipping')
    if not all(np.all(np.isfinite(a)) and np.all(a<=gates['retained_subspace_moments']['threshold'])
               for a in retained.values()):
        raise ValueError('GATE scalar_head_retained_moments: shared retained-space identity failed')
    factors,poles,active = model
    poles = poles[0,active[0]]
    factors = factors[0,0,active[0]]
    omega = np.sqrt(poles)
    residues = np.abs(factors)**2/(2*omega)
    return dict(Omega_p=omega.astype(np.complex128),B_p=residues.astype(np.complex128),
        diagnostics=dict(reduction={k:np.asarray(v).tolist() for k,v in reduction.items()},
            zero={k:np.asarray(v).tolist() for k,v in zero.items()},
            retained={k:np.asarray(v).tolist() for k,v in retained.items()}),
        provenance=dict(solve_mode='shared-Hermite-Ritz-scalar-v1',n_valid=len(omega),
            support_count=len(z),pencil_width=len(points)+1,gram_keep=gram_keep))


@lru_cache(maxsize=None)
def _scalar_check_program():
    from distrib_la import eigh_local
    from .tangential_local import _mm, _round_check_equations
    from .tangential_recipe import shared_real_pole_gates_v1_r3b as gates

    def eigh(a, *, cut=None):
        return tuple(eigh_local(a))

    @jax.jit
    def execute(omega,residue,z,values,slopes,bare,eta,m1,m3):
        model = (jnp.sqrt(2*omega*residue)[None,None],
                 omega[None]**2,jnp.ones((1,omega.size),bool))
        return _round_check_equations(model,(),(1/jnp.sqrt(bare)).reshape(1,1,1),
            values[None,:,None,None],slopes[None,:,None,None],z,eta,
            m1.reshape(1,1,1),m3.reshape(1,1,1),jnp.ones((1,1,1),jnp.complex128),
            matmul=_mm,eigh=eigh,gates=gates,ordered=False)
    return execute


def scalar_head_certificate(model,observations,*,eta_ry,static_value,inverse_m1):
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
    passive,errors,reciprocity,moments=jax.device_get(_scalar_check_program()(
        jnp.asarray(omega.real),jnp.asarray(residue.real),jnp.asarray(z),
        jnp.asarray(values),jnp.asarray(slopes),jnp.asarray(np.real(bare_coulomb)),
        jnp.asarray(eta_ry),jnp.asarray(moment_m1),jnp.asarray(moment_m3)))
    if not np.all(passive['passivity']):
        raise ValueError('GATE scalar_head_passivity: shared Coulomb-normalized bounds failed')
    if not np.isfinite(errors).all() or not np.all(reciprocity['passed']):
        raise ValueError('GATE scalar_head_samples: shared finite/reciprocity check failed')
    relative=lambda value,reference: float(abs(value-reference)/max(abs(reference),np.finfo(float).tiny))
    return dict(passivity={key:np.asarray(a).tolist() for key,a in passive.items()},
        sample_value_relative=np.asarray(errors)[0,0].tolist(),
        sample_slope_relative=np.asarray(errors)[0,1].tolist(),
        reciprocity={key:np.asarray(a).tolist() for key,a in reciprocity.items()},
        high_frequency={key:{name:np.asarray(a).tolist() for name,a in row.items()}
                        for key,row in moments.items()},
        static_relative=relative(-2*np.sum(residue/omega),static_value),
        inverse_m1_relative=relative(np.sum(residue/omega**2),inverse_m1))

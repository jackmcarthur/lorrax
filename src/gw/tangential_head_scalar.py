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

"""Charge/current cross pencils on stacks of parent-local arrays.

The joint projection is plan section 12.2. Both endpoint output panels
must be retained: C[X_C,X_T] and T[X_C,X_T]. For the ordered response the
definite member is H, not G (shared-pole report equation 5.5).
These functions run inside the constructor's batched linalg stage; they
neither move an operator to the host nor prescribe a processor mesh.
"""

import jax.numpy as jnp


def ordered_cross_pencil(charge, transverse, cross_actions, cross_moments, *, matmul):
    """Assemble rectangular G_CT, H_CT and both cross outputs, report A.1/5.4.

    ``charge`` and ``transverse`` are (nodes, directions, infinity_directions),
    with shapes [b,R_f], [b,n,R_f], [b,n,r_inf]. The finite columns already
    follow the round's paired order. ``cross_actions`` contains W_TC Q_C,
    W_CT Q_T and (dW_CT/dz) Q_T at each column's own node, with shapes
    [b,n_T,R_C], [b,n_C,R_T], [b,n_C,R_T]. ``cross_moments`` contains
    M0_CT..M3_CT [b,n_C,n_T], half the physical z-series coefficients.
    Arrays remain parent-local inside an admitted batched linalg program.

    Returns G_CT, H_CT [b,R_C+2r_C,R_T+2r_T] and the full cross output
    panels O_TC, O_CT. No adjoint symmetry is imposed on a cross tile.
    """
    from gw.shared_pole_pencil import _finite_column_g

    zc, qc, ic = charge
    zt, qt, it = transverse
    tc, ct, derivative = cross_actions
    left = matmul(tc, qt, transa="C")
    right = matmul(qc, ct, transa="C")
    g = _finite_column_g(left, right, matmul(qc, derivative, transa="C"), zc, zt)
    h = zt[:, None, :] * g - left
    mt = tuple(2 * matmul(m, it) for m in cross_moments)
    mc = tuple(2 * matmul(m, ic, transa="C") for m in cross_moments)
    # Infinity rows on the C side and columns on the T side use the
    # same resolvent identity, with each side's own complex coordinate.
    bottom0 = matmul(ic, ct, transa="C")
    bottom1 = zt[:, None, :] * bottom0 - matmul(mc[0], qt, transa="C")
    bottom2 = zt[:, None, :] * bottom1 - matmul(mc[1], qt, transa="C")
    top0 = matmul(tc, it, transa="C")
    top1 = jnp.conj(zc)[:, :, None] * top0 - matmul(qc, mt[0], transa="C")
    top2 = jnp.conj(zc)[:, :, None] * top1 - matmul(qc, mt[1], transa="C")
    p = tuple(matmul(ic, m, transa="C") for m in mt)
    block = lambda a, b, c, d: jnp.concatenate((
        jnp.concatenate((a, b), axis=-1), jnp.concatenate((c, d), axis=-1)), axis=-2)
    g = block(g, jnp.concatenate((top0, top1), axis=-1),
              jnp.concatenate((bottom0, bottom1), axis=-2), block(p[0], p[1], p[1], p[2]))
    h = block(h, jnp.concatenate((top1, top2), axis=-1),
              jnp.concatenate((bottom1, bottom2), axis=-2), block(p[1], p[2], p[2], p[3]))
    return g, h, jnp.concatenate((tc, mc[0], mc[1]), axis=-1), jnp.concatenate((ct, mt[0], mt[1]), axis=-1)


def cross_pencil_block(left, right, samples, *, matmul):
    """Compute one rectangular Hermite block, report equation A.1.

    Parameters
    ----------
    left, right : tuple
        Node and direction panel, ``(s, Q[b,n,r])``. Nodes are z² for
        the even pencil and z for the signed pencil, in Ry² and Ry.
    samples : tuple
        CT at conjugate(left node), CT at right node, and its derivative
        at right node. Arrays are complex ``[b,n_C,n_T]``. The derivative
        is with respect to the pencil coordinate. CT at the conjugate
        node must come from TC's adjoint, never from CT's own adjoint.
    matmul : callable
        Constructor's local/service GEMM with transa/transb support.

    Returns
    -------
    tuple
        Cross G and H, complex ``[b,r_C,r_T]``. Units follow the input
        state normalization. Parent sharding is inherited from the caller.
    """
    a, qc = left
    b, qt = right
    wa, wb, derivative = samples
    project = lambda w: matmul(qc, matmul(w, qt), transa="C")
    at_left = project(wa)
    delta = b - jnp.conj(a)
    confluent = delta == 0
    g = jnp.where(confluent, -project(derivative),
                  (at_left - project(wb)) / jnp.where(confluent, 1, delta))
    return g, b * g - at_left


def joint_sector_pencil(charge, transverse, cross, *, matmul):
    """Project CT on the two metric-corrected sector spans (plan 12.2).

    Parameters
    ----------
    charge, transverse : tuple
        ``(Y, values, own_output, cross_output)``. Y is ``[b,R,K]``;
        values ``[b,K]`` are squared Ritz poles for even data or signed
        inverse poles for ordered data. Output panels before projection
        have ``[b,n_endpoint,R]``. Charge cross_output has T rows;
        transverse cross_output has C rows. Y is normalized in G for
        even data, H for ordered data. Only retained columns enter.
    cross : tuple
        Cross definite and value members ``[b,R_C,R_T]``: (G_CT,H_CT)
        for even data, (H_CT,G_CT) for ordered data.
    matmul : callable
        Constructor's service/local GEMM. All arrays stay parent-local
        inside the admitted batched linalg stage.

    Returns
    -------
    tuple
        Joint definite member, value member, and the two full output
        panels. No Hermitization, clipping or Gram repair is performed.
    """
    yc, vc, oc, tc = charge
    yt, vt, ot, ct = transverse
    project = lambda a: matmul(yc, matmul(a, yt), transa="C")
    metric, value = map(project, cross)
    adj = lambda a: jnp.conj(jnp.swapaxes(a, -1, -2))
    block = lambda a, b, d: jnp.concatenate((
        jnp.concatenate((a, b), axis=-1),
        jnp.concatenate((adj(b), d), axis=-1)), axis=-2)
    # Inactive columns of a batched retained span are exactly zero. Their
    # metric is zero too; assigning them an identity invents latent states.
    ic = jnp.eye(vc.shape[-1], dtype=metric.dtype)[None] * jnp.any(yc != 0, axis=-2)[:, None, :]
    it = jnp.eye(vt.shape[-1], dtype=metric.dtype)[None] * jnp.any(yt != 0, axis=-2)[:, None, :]
    return (block(ic, metric, it),
            block(ic * vc[:, None, :], value, it * vt[:, None, :]),
            jnp.concatenate((matmul(oc, yc), matmul(ct, yt)), axis=-1),
            jnp.concatenate((matmul(tc, yc), matmul(ot, yt)), axis=-1))


def reduce_sector_pencil(pencil, *, eigh, matmul, gates):
    """Solve the definite joint pair without modifying either member.

    ``pencil`` is (metric, value, O_C, O_T), stacked on [b,...], from
    :func:`joint_sector_pencil`. The returned (c_C,c_T,lambda,active)
    evaluates as c_C (s-lambda)^-1 c_T.H for even data, and as
    c_C (z*lambda-1)^-1 c_T.H for ordered data. The latter follows
    report equation 5.5. The constructor must refuse false diagnostics.

    ``eigh`` and ``matmul`` are the admitted batched linalg callables.
    Existing normalized-Gram thresholds control rank revelation; no
    cross-sector passivity or PSD repair is applied.
    """
    from gw.shared_pole_reduction import _metric_inverse_root

    metric, value, oc, ot = pencil
    gamma, u = eigh(metric)
    top = gamma[:, -1]
    ratio = gamma[:, 0] / jnp.where(top > 0, top, 1)
    valid = ((top > 0) & jnp.all(jnp.isfinite(gamma), axis=-1)
             & (ratio >= gates["normalized_gram_validity"]["threshold"]))
    keep = ((top[:, None] > 0) &
            (gamma > gates["normalized_gram_keep"]["threshold"] * top[:, None]))
    y = u * (keep / jnp.sqrt(jnp.where(keep, gamma, 1)))[:, None, :]
    null = jnp.eye(metric.shape[-1], dtype=metric.dtype)[None] * (~keep)[:, None, :]
    reduced_metric = matmul(y, matmul(metric, y), transa="C")
    correction, corrected, diagnostics = _metric_inverse_root(
        reduced_metric + null, matmul=matmul,
        tolerance=gates["retained_subspace_moments"]["threshold"])
    y = matmul(y, correction) * keep[:, None, :]
    reduced = matmul(y, matmul(value, y), transa="C")
    sentinel = -(jnp.linalg.norm(reduced, axis=(-2, -1)) + 1)
    values, rotation = eigh(reduced + null * sentinel[:, None, None])
    count = jnp.sum(keep, axis=-1)
    active = jnp.arange(metric.shape[-1])[None] >= metric.shape[-1] - count[:, None]
    coefficients = matmul(y, rotation) * active[:, None, :]
    return (matmul(oc, coefficients), matmul(ot, coefficients),
            jnp.where(active, values, 1), active), dict(
                diagnostics, gram_valid=valid, gram_min_relative=ratio,
                retained_metric_positive=corrected, retained_rank=count)


def sector_cauchy_schwarz(metrics, *, eigh_charge, eigh_current, matmul, gates):
    """Report the squared cross norm in a positive sector metric.

    ``metrics=(C,CT,T)`` consists of [b,n_C,n_C], [b,n_C,n_T], and
    [b,n_T,n_T] Hermitian diagonal metrics and a rectangular cross block,
    for example the positive spectral moment. Cauchy--Schwarz is
    ||C^(-1/2) CT T^(-1/2)||_2² <= 1. This is not an inequality on an
    arbitrary complex-frequency W tile. No input is repaired or replaced.
    A cross block outside either metric support is reported as infinity.
    Service eigenplans and the caller's parent-local GEMM own all algebra.
    """
    c, cross, t = metrics
    cutoff = gates["normalized_gram_keep"]["threshold"]

    def inverse_root(a, eigh):
        values, vectors = eigh(a)
        keep = values > cutoff * values[:, -1:]
        scale = keep / jnp.sqrt(jnp.where(keep, values, 1))
        return (matmul(vectors * scale[:, None, :], vectors, transb="C"),
                matmul(vectors * keep[:, None, :], vectors, transb="C"),
                values[:, 0])

    ic, pc, min_c = inverse_root(c, eigh_charge)
    it, pt, min_t = inverse_root(t, eigh_current)
    whitened = matmul(ic, matmul(cross, it))
    eigenvalues, _ = eigh_charge(matmul(whitened, whitened, transb="C"))
    norm = jnp.linalg.norm(cross, axis=(-2, -1))
    outside = jnp.linalg.norm(cross-matmul(pc, matmul(cross, pt)), axis=(-2, -1))
    defect = outside / jnp.maximum(norm, jnp.finfo(norm.dtype).tiny)
    supported = defect <= gates["retained_subspace_moments"]["threshold"]
    return dict(cauchy_schwarz_squared=jnp.where(supported, eigenvalues[:, -1], jnp.inf),
                support_relative=defect, charge_metric_min=min_c, current_metric_min=min_t)

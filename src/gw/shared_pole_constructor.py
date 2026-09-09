"""Tangential Hermite/Ritz construction of physical shared real-pole W.

The physical convention is W(s) = C (s - Lambda)^-1 C.H, s = z_Ry**2;
S_m = 2 M_(2m+1).  Thus C has units Ry**(3/2), Lambda Ry**2, and
M1/M3 are physical moments in Ry**3/Ry**5, never bare chi coefficients.

Sample matrices are consumed in bounded batches.  Only narrow direction,
output and derivative-action panels survive to the pencil assembly.  Dense
products and eigensolves are supplied by the resolved distrib_la operations;
there is no local vendor or alternative eigensolver in this physics owner.
"""

from __future__ import annotations

import jax.numpy as jnp


def _adjoint(a):
    return jnp.conj(jnp.swapaxes(a, -1, -2))


def _hermitian(a):
    return (a + _adjoint(a)) * 0.5


def finite_pencil_column(left, right, *, matmul):
    """Form one block column of the resolvent-identity pencil.

    Parameters
    ----------
    left : tuple
        ``(s, Q, O)`` with s [R] in Ry**2, Q/O [b,n,R] complex128
        face tiles and O_a = W(s_a) Q_a. Repeated entries of s label
        separate tangential columns, including conjugate and role states.
    right : tuple
        ``(s_b, Q_b, O_b, D_b)`` with panels [b,n,r] in the same layout,
        O_b = W(s_b) Q_b and D_b = dW(s_b)/ds Q_b.
    matmul : callable
        Resolved service GEMM; accepts ``transa='C'`` for the adjoint.

    Returns
    -------
    g, h : arrays
        [b,R,r] complex128 face tiles. They are X.H X and X.H T X
        blocks for X_b = (s_b-T)^-1 C.H Q_b (algorithm guide, section 4).
    """
    sa, qa, oa = left
    sb, qb, ob, db = right
    a = matmul(oa, qb, transa="C")
    b = matmul(qa, ob, transa="C")
    derivative = matmul(qa, db, transa="C")
    denominator = sb - jnp.conj(sa)
    # This is the inherited floating-point equality test for confluent s,
    # not a physical support-merging tolerance. Roles are never merged.
    scale = jnp.maximum(1.0, jnp.maximum(jnp.abs(sa), jnp.abs(sb)))
    confluent = jnp.abs(denominator) <= 8 * jnp.finfo(jnp.float64).eps * scale
    safe = jnp.where(confluent, 1.0 + 0j, denominator)
    g = jnp.where(confluent[None, :, None], -derivative,
                  (a - b) / safe[None, :, None])
    return g, sb * g - a


def infinity_pencil_column(finite, infinity, *, matmul):
    """Form infinity rows using physical M1 and M3 (guide section 4).

    ``finite=(s,Q,O)`` carries [R], [b,n,R], [b,n,R].
    ``infinity=(Q_inf,M1_Q_inf,M3_Q_inf)`` carries three [b,n,r_inf]
    complex128 face panels. Returns G_inf,finite, H_inf,finite,
    G_inf,inf, H_inf,inf and O_inf. No full moment matrix is retained.
    """
    s, q, output = finite
    qi, m1qi, m3qi = infinity
    gi = matmul(qi, output, transa="C")
    hi = gi * s[None, None, :] - 2 * matmul(m1qi, q, transa="C")
    gii = 2 * matmul(qi, m1qi, transa="C")
    hii = 2 * matmul(qi, m3qi, transa="C")
    return gi, hi, gii, hii, 2 * m1qi


def assemble_shared_pole_pencil(states, infinity, *, matmul):
    """Assemble G, H and O from bounded-sample action panels.

    ``states`` is an ordered sequence of ``(s,Q,WQ,dWQ)`` tuples, where
    s is a scalar Ry**2 and panels are complex128 [b,n,r_a] face tiles.
    Each state preserves its own role, even when samples are shared.
    ``infinity`` contains Q_inf, M1 Q_inf, M3 Q_inf [b,n,r_inf].
    Only narrow panels and the dense pencil are resident here; full W
    samples must already have been released by their bounded producer.
    """
    q = jnp.concatenate([state[1] for state in states], axis=-1)
    output = jnp.concatenate([state[2] for state in states], axis=-1)
    s = jnp.concatenate([jnp.full(state[1].shape[-1], state[0], jnp.complex128)
                         for state in states])
    finite = (s, q, output)
    columns = [finite_pencil_column(finite, state, matmul=matmul) for state in states]
    g = jnp.concatenate([column[0] for column in columns], axis=-1)
    h = jnp.concatenate([column[1] for column in columns], axis=-1)
    gi, hi, gii, hii, oi = infinity_pencil_column(finite, infinity, matmul=matmul)
    g = jnp.concatenate((jnp.concatenate((g, _adjoint(gi)), axis=-1),
                         jnp.concatenate((gi, gii), axis=-1)), axis=-2)
    h = jnp.concatenate((jnp.concatenate((h, _adjoint(hi)), axis=-1),
                         jnp.concatenate((hi, hii), axis=-1)), axis=-2)
    return _hermitian(g), _hermitian(h), jnp.concatenate((output, oi), axis=-1)


def reduce_shared_pole_pencil(pencil, active_columns, *, eigh, matmul, gates):
    """Equilibrate the Gram matrix and compute its corrected Ritz model.

    Parameters
    ----------
    pencil : tuple
        G/H [b,R,R] and O [b,n,R], complex128 face tiles. G=X.H X,
        H=X.H T X, O=C_exact X. R includes inert mesh-padding columns.
    active_columns : array
        [b,R] boolean replicated mask; only declared padding is inactive.
    eigh : callable
        The common service Plan.batched Hermitian eigensolve, returning
        replicated eigenvalues and face-tiled column eigenvectors.
    matmul : callable
        Resolved service GEMM with adjoint support.
    gates : mapping
        Canonical ``shared_real_pole_gates_v1_r3b`` table from the input owner.

    Returns
    -------
    model : tuple
        C [b,n,R], poles2 [b,R], active [b,R]. Inactive poles use 1 Ry**2.
        Columns have not yet been compacted into the active prefix.
    diagnostics : mapping
        Device-resident predicates and scalars. Caller evaluates all ranks'
        reductions before formatting and refuses failed predicates.
    coefficients : array
        [b,R,R] physical Ritz coefficient map Y, including equilibration and
        rotation: C=O_original Y and Y.H G_original Y=diag(active).
        Returned for retained-space diagnostics, never a frozen SC basis.
    """
    g, h, output = pencil
    diagonal = jnp.real(jnp.diagonal(g, axis1=-2, axis2=-1))
    diagonal_ok = jnp.all(jnp.where(active_columns,
                                  jnp.isfinite(diagonal) & (diagonal > 0),
                                  diagonal == 0), axis=-1)
    # Invalid physical diagonals have a finite arithmetic placeholder only;
    # diagonal_ok is a compulsory refusal predicate, never a diagonal floor.
    scale = jnp.where(active_columns,
                      1 / jnp.sqrt(jnp.where(diagonal > 0, diagonal, 1)), 0)
    g = scale[:, :, None] * g * scale[:, None, :]
    h = scale[:, :, None] * h * scale[:, None, :]
    output = output * scale[:, None, :]
    gamma, u = eigh(_hermitian(g))
    largest = gamma[:, -1]
    ratio = gamma[:, 0] / jnp.where(largest > 0, largest, 1)
    gram_ok = ((largest > 0) & jnp.all(jnp.isfinite(gamma), axis=-1)
               & (ratio >= gates["normalized_gram_validity"]["threshold"]))
    keep = gamma > gates["normalized_gram_keep"]["threshold"] * largest[:, None]
    keep = keep & (largest[:, None] > 0)
    count = jnp.sum(keep, axis=-1, dtype=jnp.int64)
    z = u * (keep / jnp.sqrt(jnp.where(keep, gamma, 1)))[:, None, :]

    metric = matmul(z, matmul(g, z), transa="C")
    null_identity = jnp.eye(g.shape[-1], dtype=g.dtype)[None] * (~keep)[:, None, :]
    metric_eigen, metric_vectors = eigh(_hermitian(metric) + null_identity)
    metric_ok = jnp.all(jnp.isfinite(metric_eigen) & (metric_eigen > 0), axis=-1)
    correction = matmul(
        metric_vectors / jnp.sqrt(jnp.where(metric_eigen > 0, metric_eigen, 1))[:, None, :],
        metric_vectors, transb="C")
    z = matmul(z, correction) * keep[:, None, :]
    metric = matmul(z, matmul(g, z), transa="C")
    t = _hermitian(matmul(z, matmul(h, z), transa="C"))
    # The norm puts the inert spectrum strictly below every physical Ritz
    # value, including negative physical values which must reach zero policy.
    sentinel = -(jnp.linalg.norm(t, axis=(-2, -1)) + 1)
    t = t + null_identity * sentinel[:, None, None]
    poles, rotation = eigh(t)
    active = jnp.arange(g.shape[-1])[None, :] >= g.shape[-1] - count[:, None]
    c = matmul(matmul(output, z), rotation) * active[:, None, :]
    poles = jnp.where(active, poles, 1.0)
    wanted_metric = jnp.eye(g.shape[-1], dtype=g.dtype)[None] * keep[:, None, :]
    diagnostics = {
        "gram_diagonal_positive": diagonal_ok,
        "gram_valid": gram_ok,
        "gram_min_relative": ratio,
        "retained_rank": count,
        "gram_condition": largest / jnp.min(jnp.where(keep, gamma, jnp.inf), axis=-1),
        "retained_metric_positive": metric_ok,
        "retained_metric_relative": jnp.linalg.norm(metric - wanted_metric, axis=(-2, -1))
        / jnp.sqrt(jnp.maximum(count, 1)),
    }
    # Undo equilibration in the coefficient map used by retained-space checks.
    coefficients = matmul(z, rotation) * active[:, None, :]
    return (c, poles, active), diagnostics, scale[:, :, None] * coefficients


def retained_moment_identity(pencil, coefficients, model, infinity_selector, *, matmul):
    """Check the two moment blocks on projected latent infinity states.

    Let E select infinity columns of X, G=X.H X, H=X.H T X, and Y be
    the active Ritz coefficient map after both Gram and zero-policy cuts.
    A=Y.H G E and B=Y A represent P_ret X_inf = X B. The independent
    original-pencil targets are B.H G B / 2 and B.H H B / 2; the model
    values are A.H A / 2 and A.H Lambda A / 2. These compare the same
    retained latent states, not the discarded original infinity components.

    ``pencil=(G,H,O)`` has face tiles [b,R,R]/[b,n,R]; coefficients is
    [b,R,Kp], model=(C,Lambda,active), and infinity_selector E [b,R,r_inf].
    All arrays are complex128 face tiles except replicated real Lambda and
    boolean active. Returns relative Frobenius defects, one per q and moment.
    """
    g, h, _ = pencil
    _, poles, active = model
    y = coefficients * active[:, None, :]
    a = matmul(y, matmul(g, infinity_selector), transa="C")
    projected = matmul(y, a)
    rows = {}
    for name, operator, weight in (("M1", g, jnp.ones_like(poles)), ("M3", h, poles)):
        exact = matmul(projected, matmul(operator, projected), transa="C") / 2
        reconstructed = matmul(a, a * weight[:, :, None], transa="C") / 2
        norm = jnp.linalg.norm(exact, axis=(-2, -1))
        defect = jnp.linalg.norm(reconstructed - exact, axis=(-2, -1))
        rows[name] = jnp.where(norm > 0, defect / jnp.where(norm > 0, norm, 1),
                               jnp.where(defect == 0, 0, jnp.inf))
    return rows


def apply_shared_pole_zero_policy(model, *, gates):
    """Drop low Ritz values only within the resolved factor-weight budget.

    ``model=(C,poles2,active)`` has shapes [b,n,Kp], [b,Kp], [b,Kp].
    The weight is sum_j ||C_j||**2 (Ry**3), excluding carrier sentinels.
    A failed predicate must refuse before export; no pole is clipped.
    """
    c, poles, active = model
    drop = active & (poles <= gates["zero_ritz_policy"]["threshold"]["lambda_cutoff_ry2"])
    keep = active & ~drop
    weights = jnp.sum(jnp.abs(c) ** 2, axis=-2)
    total = jnp.sum(jnp.where(active, weights, 0), axis=-1)
    lost = jnp.sum(jnp.where(drop, weights, 0), axis=-1)
    fraction = lost / jnp.where(total > 0, total, 1)
    count = jnp.sum(keep, axis=-1, dtype=jnp.int64)
    finite = jnp.all(jnp.isfinite(c), axis=(-2, -1)) & jnp.all(jnp.isfinite(poles), axis=-1)
    admitted = (finite & (total > 0) & (count > 0)
                & (fraction <= gates["zero_ritz_policy"]["threshold"]["max_dropped_weight_fraction"]))
    return (jnp.where(keep[:, None, :], c, 0), jnp.where(keep, poles, 1), keep), {
        "zero_policy": admitted,
        "dropped_factor_weight_fraction": fraction,
        "dropped_count": jnp.sum(drop, axis=-1, dtype=jnp.int64),
        "factor_weight": total,
        "retained_rank": count,
    }


def sort_shared_pole_columns(model):
    """Sort joint active C/Lambda columns, retaining ties and safe padding.

    Returns the same three model arrays and the replicated [b,Kp]
    permutation. Active entries form a prefix, inactive C is exactly zero,
    inactive Lambda is 1 Ry**2. Stable sorting preserves equal-pole order.
    """
    c, poles, active = model
    order = jnp.argsort(jnp.where(active, poles, jnp.inf), axis=-1, stable=True)
    c = jnp.take_along_axis(c, order[:, None, :], axis=-1)
    poles = jnp.take_along_axis(poles, order, axis=-1)
    active = jnp.take_along_axis(active, order, axis=-1)
    return (jnp.where(active[:, None, :], c, 0), jnp.where(active, poles, 1), active), order


def shared_pole_passivity(model, inverse_coulomb_sqrt, *, eta_ry, matmul, eigh, gates):
    """Test 0 <= V^-1/2 [-W(i eta)] V^-1/2 <= I on Coulomb support.

    The authenticated inverse square root [b,n,n] and C [b,n,Kp] are
    face-tiled complex128. Lambda/active [b,Kp] are replicated. Returns
    device scalars; all ranks must evaluate them before the host refusal.
    """
    c, poles, active = model
    whitened = matmul(inverse_coulomb_sqrt, c)
    weight = jnp.where(active, 1 / (poles + eta_ry**2), 0)
    response = matmul(whitened * weight[:, None, :], whitened, transb="C")
    herm = _hermitian(response)
    norm = jnp.linalg.norm(herm, axis=(-2, -1))
    anti = jnp.linalg.norm(response - _adjoint(response), axis=(-2, -1))
    anti = anti / jnp.where(norm > 0, 2 * norm, 1)
    values, _ = eigh(herm)
    minimum, maximum = values[:, 0], values[:, -1]
    passed = ((minimum >= gates["passivity"]["threshold"]["eigenvalue_min"])
              & (maximum <= gates["passivity"]["threshold"]["eigenvalue_max"])
              & (anti <= gates["passivity"]["threshold"]["antihermitian_relative_max"])
              & jnp.all(jnp.isfinite(values), axis=-1) & jnp.isfinite(anti))
    return {"passivity": passed, "passivity_min": minimum,
            "passivity_max": maximum, "passivity_antihermitian_relative": anti}

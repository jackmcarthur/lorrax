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

from functools import lru_cache

import jax.numpy as jnp


def shared_pole_byte_terms(meta, *, mesh_xy, resolution, pencil_side,
                           parent_batch, sample_batch):
    """Price constructor carriers; the map CapacityLedger owns admission.

    Bytes include the actual parent/sample batch, narrow retained actions,
    original and equilibrated pencils, corrected Ritz eigensolver outputs,
    sort temporaries and the 2n SVD dilation. Native workspace is separately
    supplied by the service. No U threshold or independent capacity policy
    lives in this constructor helper.
    """
    import math

    p = int(mesh_xy.shape["x"]) * int(mesh_xy.shape["y"])
    packed = int(meta.n_rmu_padded) * int(meta.nspinor)
    b, a, r = int(parent_batch), int(sample_batch), int(pencil_side)
    if min(packed, b, a) <= 0 or r < 0:
        raise ValueError("GATE shared_pole_capacity: got: invalid extents; want: positive basis/batches and nonnegative pencil; why: live-set pricing")
    dense_copies = math.ceil(b / p) if resolution.layout == "local" else b / p
    directions = math.ceil(16 * dense_copies * (24 * packed**2 + 12 * packed * r))
    reduction = math.ceil(16 * dense_copies * (14 * r*r + 12 * packed * r))
    terms = {
        "sample_and_moment_batch": math.ceil(16*b*(2*a+2)*packed**2/p),
        "narrow_actions": math.ceil(16*b*6*packed*r/p),
        "replicated_scalars": 8*b*(12*r+4*packed),
        "largest_dense_phase": max(directions, reduction),
    }
    return {"terms_bytes_per_rank": terms,
            "resident_bytes_per_rank": sum(terms.values()),
            "layout": resolution.layout, "pencil_side": r,
            "parent_batch": b, "sample_batch": a}


def _adjoint(a):
    return jnp.conj(jnp.swapaxes(a, -1, -2))


def _hermitian(a):
    return (a + _adjoint(a)) * 0.5


def _diagonal_face(values, matrix):
    """Materialize a diagonal directly in the supplied pencil face layout."""
    import jax
    return jax.jit(lambda d: jnp.eye(d.shape[-1], dtype=matrix.dtype)[None]
                   * d[:, None, :], out_shardings=matrix.sharding)(values)


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
    null_identity = _diagonal_face(~keep, g)
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
    wanted_metric = _diagonal_face(keep, g)
    diagnostics = {
        "gram_diagonal_positive": diagonal_ok,
        "gram_valid": gram_ok,
        "gram_min_relative": ratio,
        "gram_spectrum_relative": gamma / jnp.where(largest > 0, largest, 1)[:, None],
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


@lru_cache(maxsize=None)
def _factor_column_permutation(mesh):
    """Stream a joint pole-column permutation through one y tile at a time.

    A global gather of C replicates the complete K axis on each x row.
    Instead, each y shard circulates its input tile and selects only the
    columns belonging to its output tile. The scan carries one input tile
    and one output tile; neither grows with the number of y shards.
    """
    from functools import partial
    import jax
    from jax.sharding import PartitionSpec as P
    from common.shard_map import shard_map

    ny = int(mesh.shape["y"])
    neighbors = tuple((i, (i+1) % ny) for i in range(ny))

    @jax.jit
    @partial(shard_map, mesh=mesh,
             in_specs=(P(None, "x", "y"), P(None, "y")),
             out_specs=P(None, "x", "y"), check_vma=False)
    def permute(factor, order):
        width = factor.shape[-1]
        def visit(carry, _):
            tile, output, source = carry
            local = order - source * width
            selected = jnp.take_along_axis(
                tile, jnp.clip(local, 0, width-1)[:, None, :], axis=-1)
            belongs = (local >= 0) & (local < width)
            output = jnp.where(belongs[:, None, :], selected, output)
            tile = jax.lax.ppermute(tile, "y", neighbors)
            return (tile, output, (source-1) % ny), None
        (_, output, _), _ = jax.lax.scan(
            visit, (factor, jnp.zeros_like(factor), jax.lax.axis_index("y")),
            None, length=ny)
        return output
    return permute


def sort_shared_pole_columns(model, *, mesh_xy):
    """Sort joint active C/Lambda columns, retaining ties and safe padding.

    Returns the same three model arrays and the replicated [b,Kp]
    permutation. Active entries form a prefix, inactive C is exactly zero,
    inactive Lambda is 1 Ry**2. Stable sorting preserves equal-pole order.
    """
    c, poles, active = model
    order = jnp.argsort(jnp.where(active, poles, jnp.inf), axis=-1, stable=True)
    c = _factor_column_permutation(mesh_xy)(c, order)
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


def _sample_point(recipe, sample_id):
    """Read a deduplicated physical point from the canonical role arrays."""
    import numpy as np
    rows = np.flatnonzero(np.asarray(recipe["distinct_id"]) == sample_id)
    points = [complex(value["real"], value["imag"]) if isinstance(value, dict)
              else complex(value) for value in recipe["z_ry"]]
    values = np.asarray(points)[rows]
    if not len(rows) or not np.all(values == values[0]):
        raise ValueError("GATE shared_pole_sample_identity: got: absent/inconsistent point; want: one z per distinct_id; why: roles share bank evaluations only")
    return complex(values[0])


def _fit_roles(recipe):
    """Ephemeral constructor record view; only flat arrays are serialized."""
    from gw.shared_pole_recipe import ROLE_CODES
    names = {code: name for name, code in ROLE_CODES.items()}
    return [{"sample_id": int(sample), "role": f"{names[int(role)]}:{i}",
             "held": bool(held)}
            for i, (sample, role, held) in enumerate(zip(
                recipe["distinct_id"], recipe["role"], recipe["held"]))]


def _direction_states(read_sample, recipe, *, eigh_plan, svd_plan, matmul,
                      column_extent, logical_n, admit, infinity_carrier):
    """Read each distinct fitted sample once, preserving all tangent roles."""
    import distrib_la

    fit_roles = _fit_roles(recipe)
    states, masks, roles = [], [], []
    for sample_id in recipe["fit_ids"]:
        # Admit before reading/selecting, including the largest possible
        # next multiplet and all conjugate/duplicate roles of this sample.
        next_roles = [role for role in fit_roles
                      if not role["held"] and int(role["sample_id"]) == int(sample_id)]
        maximum_new = sum(2 if role["role"].startswith("line:") else 1
                          for role in next_roles) * column_extent(logical_n)
        admit(infinity_carrier + sum(state[1].shape[-1] for state in states) + maximum_new)
        w, derivative = read_sample(int(sample_id), states)
        for role in fit_roles:
            if role["held"] or int(role["sample_id"]) != int(sample_id):
                continue
            kind = role["role"].split(":", 1)[0]
            if kind == "line":
                q, values = distrib_la.right_singular_vectors(
                    w[0], recipe["direction_cutoff"], eigh_plan=svd_plan,
                    column_extent=column_extent,
                    multiplet_tol=recipe["multiplet_relative_tolerance"])
            elif kind == "imaginary":
                width = min(logical_n, max(1, int(recipe["imaginary_width"])))
                q, values = distrib_la.leading_eigenvectors(
                    -w[0], width, eigh_plan=eigh_plan, column_extent=column_extent,
                    multiplet_tol=recipe["multiplet_relative_tolerance"])
            else:
                raise ValueError(f"GATE shared_pole_role: got: {kind}; want: line or imaginary fitted role; why: unknown tangent semantics")
            width = int(values.shape[-1])
            if width < 1 or width > logical_n:
                raise ValueError(f"GATE shared_pole_directions: got: rank {width}; want: 1..{logical_n}; why: empty or padded physical direction set")
            q = q[None]
            s = _sample_point(recipe, int(sample_id)) ** 2
            # The line's conjugate state reuses the SAME right directions,
            # exactly as the latent Hermite construction specifies. No extra
            # sample and no re-selection on the adjoint matrix is performed.
            for conjugate in ((False, True) if kind == "line" and s.imag != 0 else (False,)):
                transa = "C" if conjugate else "N"
                output = matmul(w, q, transa=transa)
                action = matmul(derivative, q, transa=transa)
                states.append((s.conjugate() if conjugate else s, q, output, action))
                masks.append(jnp.arange(q.shape[-1]) < width)
                roles.append({"sample_id": int(sample_id), "role": role["role"],
                              "conjugate": conjugate, "width": width,
                              "carrier_width": q.shape[-1]})
        del w, derivative
    return states, masks, roles


def _model_diagnostics(model, moments, infinity_directions, *, matmul):
    """Full and original-infinity projected physical moment diagnostics."""
    c, poles, _ = model
    result = {}
    for name, target, weighted in (("M1", moments["M1"], c),
                                   ("M3", moments["M3"], c * poles[:, None, :])):
        value = matmul(weighted, c, transb="C") / 2
        defect = target - value
        projected = matmul(infinity_directions,
                           matmul(defect, infinity_directions), transa="C")
        projected_target = matmul(infinity_directions,
                                  matmul(target, infinity_directions), transa="C")
        norm = jnp.linalg.norm(target, axis=(-2, -1))
        projected_norm = jnp.linalg.norm(projected_target, axis=(-2, -1))
        result[name] = {
            "full_relative": jnp.linalg.norm(defect, axis=(-2, -1)) / norm,
            "original_infinity_relative": jnp.linalg.norm(projected, axis=(-2, -1)) / projected_norm,
        }
    return result


def construct_shared_poles(bank, moments, meta, config, *, mesh_xy, output):
    """Construct and write a current-state, bounded-batch real-pole model.

    Parameters
    ----------
    bank : mapping
        Authenticated resource descriptor: ``path``, ``identity``, ``tables``,
        ``coulomb`` (response-owner authenticated Coulomb resource). Dense
        workspace is queried from the resolved service plans. A producer
        certificate is carried as ``rule_receipt``. No dense bank is passed
        as a jit argument. The recipe is read only from meta.
    moments : mapping
        ``path`` to committed physical M1/M3 in the same scratch schema.
    meta : Meta
        Current packed centroid basis, scalar representation and the once-
        resolved ``shared_pole_recipe`` with current state identities.
    config : LorraxConfig
        Cached dense policy is read once, before any per-q plans.
    mesh_xy : Mesh
        Supplied named x/y mesh, never reconstructed by this driver.
    output : path-like
        New immutable compact model file, written by the store owner.

    Returns
    -------
    dict
        Versioned construction receipt with all gate rows, per-q diagnostics,
        capacity prices and the store header/digest. Structural failures raise
        before the affected q is written; a partial file is never finalized.
    """
    import numpy as np
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    import distrib_la
    from runtime.padding import padded_axis
    from file_io.slab_io import SlabIO
    from file_io.shared_pole_store import (
        validate_shared_pole_bank, read_shared_pole_bank, write_shared_pole_model,
    )
    from gw.gw_config import linalg_resolution
    from gw.shared_pole_recipe import (
        construction_receipt, shared_real_pole_gates_v1_r3b as gates,
    )
    from common.units import RYD_TO_EV
    from gw.w_isdf import response_coulomb_powers

    recipe = meta.shared_pole_recipe
    if int(meta.nspinor) != 1 or not bool(bank["tables"]["sym"].trs_allowed):
        raise ValueError("GATE shared_pole_representation: got: non-scalar or TRS-broken state; want: scalar with authenticated TRS; why: shared even-s representation")
    resolution = linalg_resolution({"linalg": config.backend.linalg})
    identity = bank["identity"]
    ledger = meta.shared_pole_capacity
    upstream = ledger.live_stages
    n = int(meta.n_rmu_padded)
    native_queries = {}
    native_maxima = {"eigh": 0, "gemm": 0}
    workspace = 0
    current_side = 0
    plans = {}

    def eigenplan(side):
        if side not in plans:
            plans[side] = distrib_la.plan(
                "eigh", mesh_xy, n=side, backend=resolution.eigh_backend,
                batched_route=resolution.batched_route)
        return plans[side]

    def query_workspace(op, shapes, plan):
        nonlocal workspace
        key = (op, shapes)
        if key not in native_queries:
            size = distrib_la.workspace_bytes_per_rank(
                plan, op, shapes, np.complex128)
            native_queries[key] = size
            native_maxima[op] = max(native_maxima[op], size)
            # GEMM retains its workspace in the context while eigh uses
            # transient scratch. They coexist; the provider says to sum.
            workspace = sum(native_maxima.values())
        return native_queries[key]
    header = validate_shared_pole_bank(bank["path"], expected_identity=identity,
                                       mesh_xy=mesh_xy, require_complete=True)
    moment_header = validate_shared_pole_bank(moments["path"], expected_identity=identity,
                                              mesh_xy=mesh_xy)
    # The stored plan must bind the current physical points and role census.
    stored_recipe = header["recipe"]
    for name in ("recipe_hash", "gate_hash", "fit_ids", "held_ids", "role", "role_codes",
                 "distinct_id", "held", "support_pair", "census"):
        current = recipe[name]
        previous = stored_recipe[name]
        if isinstance(current, np.ndarray):
            equal = np.array_equal(current, previous)
        else:
            equal = current == previous
        if not equal:
            raise ValueError(f"GATE shared_pole_bank_state: got: stale {name}; want: current resolved recipe; why: no frozen SC inputs")
    for sample_id in (*recipe["fit_ids"], *recipe["held_ids"]):
        if _sample_point(recipe, sample_id) != _sample_point(stored_recipe, sample_id):
            raise ValueError("GATE shared_pole_bank_state: got: changed z; want: current physical sample point; why: recipe hashes alone do not bind resolved points")

    def capacity(side):
        nonlocal current_side
        current_side = side
        for extent in sorted({n, 2*n, side} - {0}):
            query_workspace("eigh", ((1, extent, extent),), eigenplan(extent))
        query_workspace("gemm", ((1, n, n), (1, n, n)), eigenplan(n))
        price = shared_pole_byte_terms(
            meta, mesh_xy=mesh_xy, resolution=resolution, pencil_side=side,
            parent_batch=1, sample_batch=1)
        row = ledger.reserve(f"constructor.plan.{len(ledger.entries)}",
                             resident_bytes_per_rank=price["resident_bytes_per_rank"],
                             workspace_bytes_per_rank=workspace,
                             concurrent_with=upstream)
        return dict(row, price=price, native_workspace=dict(native_maxima))

    def expose_live(arrays):
        # Callees price only their additional allocations. Supply their
        # exact current inputs instead of the future dense-phase envelope,
        # so a store read does not count its own returned arrays twice.
        unique = {id(array): array for array in arrays}
        resident = sum(int(np.prod(array.sharding.shard_shape(array.shape)))
                       * array.dtype.itemsize for array in unique.values())
        row = ledger.reserve(f"constructor.live.{len(ledger.entries)}",
                             resident_bytes_per_rank=resident,
                             workspace_bytes_per_rank=workspace,
                             concurrent_with=upstream)
        ledger.live_stages = (*upstream, row["stage"])

    capacity(0)
    logical_n = int(meta.n_rmu)
    face = NamedSharding(mesh_xy, P(None, "x", "y"))
    public_factor = NamedSharding(mesh_xy, P(None, "x", None, "y"))
    column_extent = lambda width: padded_axis(
        width, mesh_xy, name="shared_pole_port",
        specs=((P("x", "y"), 0), (P("x", "y"), 1))).carrier
    def mm(a, b, **kwargs):
        # Query the actual effective N,N shapes before the service allocates
        # transpose staging, output or a larger persistent GEMM workspace.
        shapes = tuple(value.shape[:-2] + (value.shape[-2:][::-1]
                       if kwargs.get(trans, "N") != "N" else value.shape[-2:])
                       for value, trans in ((a, "transa"), (b, "transb")))
        previous = workspace
        query_workspace("gemm", shapes, eigenplan(n))
        if workspace != previous:
            capacity(current_side)
        return distrib_la.matmul(a, b, mesh=mesh_xy, backend="auto",
                                 batched_route=resolution.batched_route, **kwargs)

    eig, svd = eigenplan(n), eigenplan(2*n)
    receipts = []
    # One parent per iteration makes the maximum live set independent of the
    # total irreducible-q count. Store owns the ragged K census and final copy.
    for q in range(int(header["bank_shape"]["nq"])):
        span = (q, q + 1)
        expose_live(())
        with SlabIO(moments["path"], mode="r", mesh=mesh_xy) as moment_io:
            exact = read_shared_pole_bank(moment_io, span, meta=meta,
                                          header=moment_header, fields=("M1", "M3"))
        width = min(logical_n, max(1, int(recipe["infinity_width"])))
        qi, infinity_values = distrib_la.leading_eigenvectors(
            exact["M1"][0], width, eigh_plan=eig, column_extent=column_extent,
            multiplet_tol=recipe["multiplet_relative_tolerance"])
        qi = qi[None]
        infinity = (qi, mm(exact["M1"], qi), mm(exact["M3"], qi))
        del exact
        with SlabIO(bank["path"], mode="r", mesh=mesh_xy) as bank_io:
            def read_sample(sample_id, retained_states):
                expose_live((*infinity, *(panel for state in retained_states for panel in state[1:])))
                data = read_shared_pole_bank(
                    bank_io, span, meta=meta, header=header,
                    sample_span=(sample_id, sample_id + 1))
                return data["Wc"][:, 0], data["dWc_ds"][:, 0]

            states, masks, roles = _direction_states(
                read_sample, recipe, eigh_plan=eig, svd_plan=svd, matmul=mm,
                column_extent=column_extent, logical_n=logical_n, admit=capacity,
                infinity_carrier=qi.shape[-1])
        masks.append(jnp.arange(qi.shape[-1]) < infinity_values.shape[-1])
        active_columns = jnp.concatenate(masks)[None]
        price = capacity(active_columns.shape[-1])
        pencil = assemble_shared_pole_pencil(states, infinity, matmul=mm)
        del states, masks, infinity
        reduce_eigh = eigenplan(pencil[0].shape[-1])
        model, reduction, coefficients = reduce_shared_pole_pencil(
            pencil, active_columns, eigh=reduce_eigh.batched, matmul=mm, gates=gates)
        for name in ("gram_diagonal_positive", "gram_valid", "retained_metric_positive"):
            if not bool(jnp.all(reduction[name])):
                raise ValueError(f"GATE shared_pole_{name}: got: failed at q={q}; want: valid Gram and retained metric; why: no PSD repair")
        model, zero = apply_shared_pole_zero_policy(model, gates=gates)
        if not bool(jnp.all(zero["zero_policy"])):
            raise ValueError(f"GATE shared_pole_zero_ritz: got: failed at q={q}; want: finite positive response within dropped-weight budget; why: no pole clipping")
        # E selects the last infinity block of X. Build it as a face array;
        # only the small row/column coordinate vectors are replicated.
        r, ri = pencil[0].shape[-1], qi.shape[-1]
        selector = jax.jit(lambda: (jnp.arange(r)[:, None] ==
                                   jnp.arange(r-ri, r)[None, :])[None].astype(jnp.complex128),
                           out_shardings=face)()
        retained = retained_moment_identity(pencil, coefficients, model, selector, matmul=mm)
        if not all(bool(jnp.all(value <= gates["retained_subspace_moments"]["threshold"]))
                   for value in retained.values()):
            raise ValueError(f"GATE shared_pole_retained_moments: got: failed at q={q}; want: projected latent moment identity <=1e-10; why: corrected Ritz algebra")
        del pencil, coefficients, selector, active_columns
        model, permutation = sort_shared_pole_columns(model, mesh_xy=mesh_xy)
        expose_live((*model, qi))
        coulomb_sqrt, inverse_sqrt, coulomb_receipt = response_coulomb_powers(
            meta, config, mesh_xy=mesh_xy, bank_io=bank, q_span=span)
        passive = shared_pole_passivity(model, inverse_sqrt,
                                       eta_ry=recipe["eta_ev"] / RYD_TO_EV,
                                       matmul=mm, eigh=eig.batched, gates=gates)
        if not bool(jnp.all(passive["passivity"])):
            raise ValueError(f"GATE shared_pole_passivity: got: failed at q={q}; want: 0 <= V-whitened -W(i eta) <= I; why: passive screening")
        del coulomb_sqrt, inverse_sqrt
        expose_live((*model, qi))
        with SlabIO(moments["path"], mode="r", mesh=mesh_xy) as moment_io:
            exact = read_shared_pole_bank(moment_io, span, meta=meta,
                                          header=moment_header, fields=("M1", "M3"))
        moment_defects = _model_diagnostics(model, exact, qi, matmul=mm)
        del exact, qi
        held = []
        with SlabIO(bank["path"], mode="r", mesh=mesh_xy) as bank_io:
            for sample_id in recipe["held_ids"]:
                expose_live(model)
                samples = read_shared_pole_bank(bank_io, span, meta=meta, header=header,
                                               sample_span=(int(sample_id), int(sample_id)+1),
                                               fields=("Wc", "dWc_ds"))
                c, poles, mask = model
                s = _sample_point(recipe, int(sample_id)) ** 2
                weights = jnp.where(mask, 1 / (s-poles), 0)
                diagnostic = {"sample_id": int(sample_id)}
                for field, weight in (("Wc", weights), ("dWc_ds", -weights**2)):
                    sample = samples[field][:, 0]
                    value = mm(c * weight[:, None, :], c, transb="C")
                    diagnostic[field] = float(jnp.linalg.norm(value-sample) /
                                              jnp.maximum(jnp.linalg.norm(sample), jnp.finfo(jnp.float64).tiny))
                held.append(diagnostic)
                del samples, sample, value
        c, poles, mask = model
        counts = jnp.sum(mask, axis=-1, dtype=jnp.int64)
        # All scalar reductions precede rank-selective store formatting.
        price = capacity(r)
        row = {"q_span": list(span), "roles": roles,
               "K": np.asarray(counts).tolist(), "J": int(np.unique(np.asarray(poles)[np.asarray(mask)]).size),
               "damping_fraction": 0.0, "capacity": price, "coulomb": coulomb_receipt,
               "condition": np.asarray(reduction["gram_condition"]).tolist(),
               "normalized_gram_spectrum": np.asarray(reduction["gram_spectrum_relative"]).tolist(),
               "native_workspace_queries": [dict(op=op, shapes=shapes, bytes_per_rank=value)
                                             for (op, shapes), value in native_queries.items()],
               "retained_moment_relative": {k: np.asarray(v).tolist() for k, v in retained.items()},
               "moment_defects": {k: {a: np.asarray(b).tolist() for a, b in v.items()}
                                  for k, v in moment_defects.items()},
               "held_W": held, "permutation": np.asarray(permutation).tolist(),
               "storage_bytes": int(counts[0]) * (16*logical_n + 8)}
        measurements = {
            "normalized_gram_keep": dict(value=int(reduction["retained_rank"][0]), passed=True, reason="normalized Gram cut, current q"),
            "normalized_gram_validity": dict(value=float(reduction["gram_min_relative"][0]), passed=True, reason="normalized Gram spectrum"),
            "zero_ritz_policy": dict(value=float(zero["dropped_factor_weight_fraction"][0]), passed=True, reason="physical factor weight, sentinels excluded"),
            "finite_factors_poles": dict(value=True, passed=True, reason="zero policy, active prefix and exact inert sentinels"),
            "passivity": dict(value={k: np.asarray(v).tolist() for k, v in passive.items() if k != "passivity"}, passed=True, reason="authenticated inverse Coulomb square root at current eta"),
            "retained_subspace_moments": dict(value=row["retained_moment_relative"], passed=True, reason="A=Y†GE, B=YA; pencil B†(G,H)B/2 versus model A†(I,Lambda)A/2"),
            "held_w": dict(value=held, passed=True, reason="held W and dW/ds diagnostics recorded; no universal acceptance threshold"),
            "full_m1_defect": dict(value=float(moment_defects["M1"]["full_relative"][0]), passed=bool(moment_defects["M1"]["full_relative"][0] <= gates["full_m1_defect"]["threshold"]), reason="physical full M1 defect; CD8 diagnostic band, never a refusal"),
            "full_m3_defect": dict(value=float(moment_defects["M3"]["full_relative"][0]), passed=bool(moment_defects["M3"]["full_relative"][0] <= gates["full_m3_defect"]["threshold"]), reason="physical full M3 defect; CD8 diagnostic band, never a refusal"),
            "representation": dict(value={"nspinor": 1, "trs_allowed": True}, passed=True, reason="current typed symmetry capability"),
            "capacity": dict(value=price, passed=True, reason="conservative aggregate constructor live-set price"),
            "sc_rebuild": dict(value=identity, passed=True, reason="current recipe/census authenticated; directions and Ritz model rebuilt"),
        }
        receipt = construction_receipt(measurements, capacity=ledger)
        receipt.update(identity=identity, constructor=row)
        public_c = jax.jit(lambda value: value[:, :, None, :], out_shardings=public_factor)(c)
        del model, c, mask
        expose_live((public_c, poles, counts))
        store_header = write_shared_pole_model(output, public_c, poles, counts,
                                               q_span=span, meta=meta, tables=bank["tables"],
                                               recipe=recipe, receipts=receipt)
        receipts.append(receipt)
        del public_c, poles, counts
        ledger.live_stages = upstream
    return {"q_receipts": receipts, "model_header": store_header,
            "capacity": ledger.receipt(),
            "identity": identity, "status": "CONSTRUCTED"}

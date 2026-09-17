"""Ritz reduction of the shared-pole pencils.

Equilibration, the normalized-Gram keep cut, the coupled Newton-Schulz metric
correction and the Hermitian Ritz step for the even route (SP 10), and the paired
basis, its second cut and the signed model for the ordered route (SP 13, SP 14).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from distrib_la import (diagonal_like, face_sharding, hermitian_part,
                        join_columns, on_face)
from gw.shared_pole_pencil import _adjoint


def _metric_inverse_root(metric, *, matmul, tolerance):
    """Correct a dimensionless Hermitian metric by coupled Newton–Schulz.

    ``metric`` is [b,R,R], complex128 in the caller's face layout. Products
    use the resolved distrib_la GEMM. Y starts at A and Z at I; the coupled
    update T=(3I-ZY)/2, Y=YT, Z=TZ computes A**(-1/2) without eigenvectors.
    The initial infinity norm bounds the spectral error. For radius d<1,
    d_next <= d**2*(3+d)/4 <= d**2; choose the entire iteration count from
    that initial bound, never from an on-device residual convergence test.
    The returned diagnostics include the measured ZAZ-I residual and guard.
    """
    identity = diagonal_like(jnp.ones(metric.shape[:1] + metric.shape[-1:]), metric)
    radius = jnp.max(jnp.sum(jnp.abs(identity - metric), axis=-1), axis=-1)
    valid = jnp.isfinite(radius) & (radius < 1)
    if not isinstance(radius, jax.core.Tracer) and not bool(jnp.all(valid)):
        raise ValueError(f"GATE shared_pole_metric_inverse_root: got: infinity norm {radius.tolist()}; want: finite norm < 1; why: Newton-Schulz convergence bound")
    # Aim near float64 roundoff, leaving the physical residual gate intact.
    target = min(float(tolerance), 32 * jnp.finfo(metric.real.dtype).eps)
    safe_radius = jnp.where(valid & (radius > target), radius, .5)
    counts = jnp.maximum(0, jnp.ceil(jnp.log2(
        jnp.log(target) / jnp.log(safe_radius)))).astype(jnp.int32)
    counts = jnp.where(valid & (radius > target), counts, 0)
    iterations = jnp.where(jnp.all(valid), jnp.max(counts), 0)

    def step(_, state):
        y, z = state
        t = (3 * identity - matmul(z, y)) * .5
        return matmul(y, t), matmul(t, z)

    _, correction = jax.lax.fori_loop(0, iterations, step, (metric, identity))
    residual = matmul(correction, matmul(metric, correction)) - identity
    absolute = jnp.linalg.norm(residual, axis=(-2, -1))
    relative = absolute / jnp.sqrt(metric.shape[-1])
    diagnostics = {
        "metric_initial_infinity_norm": radius,
        "metric_inverse_root_iterations": jnp.full(radius.shape, iterations),
        "metric_inverse_root_residual_fro": absolute,
        "metric_inverse_root_residual_relative": relative,
    }
    return correction, valid & jnp.isfinite(relative) & (relative <= tolerance), diagnostics



def _within_budget(gamma, budget):
    """The largest ``budget`` entries of each ascending spectrum row (all when None)."""
    if budget is None:
        return True
    return jnp.arange(gamma.shape[-1])[None, :] >= gamma.shape[-1] - int(budget)


def reduce_shared_pole_pencil(pencil, active_columns, *, eigh, matmul, gates, keep_budget=None):
    """Equilibrate the Gram matrix and compute its corrected Ritz model.

    Parameters
    ----------
    pencil : tuple
        G/H [b,R,R] and O [b,n,R], complex128 face tiles. G=X.H X,
        H=X.H T X, O=b_exact X. R includes inert mesh-padding columns.
    active_columns : array
        [b,R] boolean replicated mask; only declared padding is inactive.
    eigh : callable
        The common service Plan.batched Hermitian eigensolve, returning
        replicated eigenvalues and face-tiled column eigenvectors.
    matmul : callable
        Resolved service GEMM with adjoint support.
    gates : mapping
        Canonical ``shared_real_pole_gates_v1_r3b`` table from the input owner.
    keep_budget : int, optional
        The recipe's pole budget: at most this many equilibrated-Gram
        directions survive the keep cut, the largest first. The Ritz step
        then acts on that subspace, so K <= keep_budget.

    Returns
    -------
    model : tuple
        b [parent,n,R], poles2 [parent,R], active [parent,R]. Inactive poles use 1 Ry**2.
        Columns have not yet been compacted into the active prefix.
    diagnostics : mapping
        Device-resident predicates and scalars. Caller evaluates all ranks'
        reductions before formatting and refuses failed predicates.
    coefficients : array
        [b,R,R] physical Ritz coefficient map Y, including equilibration and
        rotation: b=O_original Y and Y.H G_original Y=diag(active).
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
    gamma, u = eigh(hermitian_part(g))
    largest = gamma[:, -1]
    ratio = gamma[:, 0] / jnp.where(largest > 0, largest, 1)
    gram_ok = ((largest > 0) & jnp.all(jnp.isfinite(gamma), axis=-1)
               & (ratio >= gates["normalized_gram_validity"]["threshold"]))
    keep = gamma > gates["normalized_gram_keep"]["threshold"] * largest[:, None]
    keep = keep & (largest[:, None] > 0) & _within_budget(gamma, keep_budget)
    count = jnp.sum(keep, axis=-1, dtype=jnp.int64)
    z = u * (keep / jnp.sqrt(jnp.where(keep, gamma, 1)))[:, None, :]

    metric = matmul(z, matmul(g, z), transa="C")
    null_identity = diagonal_like(~keep, g)
    correction, metric_ok, metric_diagnostics = _metric_inverse_root(
        hermitian_part(metric) + null_identity, matmul=matmul,
        tolerance=gates["retained_subspace_moments"]["threshold"])
    z = matmul(z, correction) * keep[:, None, :]
    metric = matmul(z, matmul(g, z), transa="C")
    t = hermitian_part(matmul(z, matmul(h, z), transa="C"))
    # The norm puts the inert spectrum strictly below every physical Ritz
    # value, including negative physical values which must reach zero policy.
    sentinel = -(jnp.linalg.norm(t, axis=(-2, -1)) + 1)
    t = t + null_identity * sentinel[:, None, None]
    poles, rotation = eigh(t)
    active = jnp.arange(g.shape[-1])[None, :] >= g.shape[-1] - count[:, None]
    b = matmul(matmul(output, z), rotation) * active[:, None, :]
    poles = jnp.where(active, poles, 1.0)
    wanted_metric = diagonal_like(keep, g)
    diagnostics = {
        **metric_diagnostics,
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
    return (b, poles, active), diagnostics, scale[:, :, None] * coefficients


ORIENTATION_PAIR_REFUSAL = ("GATE shared_pole_orientation_pair: got: finite columns not in mirrored halves; want: [X(z); X(-z)] on one direction set and paired k0/k1 columns; why: the ordered cut acts in the paired basis")


def _paired_member(a, inverse, *, half, finite, n_inf):
    """One pencil member [b,R,R] in the paired basis: its (ww, wv, vv) blocks."""
    def columns(x):
        w = jnp.concatenate(((x[..., :half] + x[..., half:finite]) / 2, x[..., finite + n_inf:]), axis=-1)
        v = jnp.concatenate(((x[..., :half] - x[..., half:finite]) * inverse[:, None, :],
                             x[..., finite:finite + n_inf]), axis=-1)
        return w, v

    def rows(x):
        w = jnp.concatenate(((x[:, :half] + x[:, half:finite]) / 2, x[:, finite + n_inf:]), axis=1)
        v = jnp.concatenate((jnp.conj(inverse)[:, :, None] * (x[:, :half] - x[:, half:finite]),
                             x[:, finite:finite + n_inf]), axis=1)
        return w, v

    w, v = columns(a)
    ww, _ = rows(w)
    wv, vv = rows(v)
    return ww, wv, vv


def _paired_output(a, inverse, *, half, finite, n_inf):
    """Output columns O [b,n,R] in the paired basis: (w, v)."""
    w = jnp.concatenate(((a[..., :half] + a[..., half:finite]) / 2, a[..., finite + n_inf:]), axis=-1)
    v = jnp.concatenate(((a[..., :half] - a[..., half:finite]) * inverse[:, None, :],
                         a[..., finite:finite + n_inf]), axis=-1)
    return w, v


def _restricted_block(ww, wv, vv):
    """Hermitian [[ww, wv], [wv^H, vv]] of the restricted paired pencil."""
    return hermitian_part(jnp.concatenate(
        (jnp.concatenate((ww, wv), axis=-1), jnp.concatenate((_adjoint(wv), vv), axis=-1)), axis=-2))


def reduce_ordered_shared_pole_pencil(pencil, active_columns, *, eigh, matmul, gates, keep_budget=None):
    """Paired-basis Ritz reduction of the particle-hole pencil.

    ``pencil=(G,H,O,z)`` from ``assemble_ordered_shared_pole_pencil`` over paired
    states: finite columns [X(z_b) ; X(-z_b)] on the same directions, then the
    infinity columns [k0 ; k1]. The congruence w_b = (X(z_b)+X(-z_b))/2,
    v_b = (X(z_b)-X(-z_b))/(2 z_b), w_inf = k1, v_inf = k0 gives H' and G'. On
    time-reversal-symmetric data H' = diag(H_s, G_s) and G' = offdiag(G_s), with
    (G_s, H_s) the even s-pencil. Equilibration, keep cut and metric correction
    therefore act on H'_vv alone, as the even route acts on G_s, and the kept
    span Z is applied to both halves. The restricted pencil has
    H_r = [[A, B], [B^H, I]] = L L^H, L = [[S^1/2, B], [0, I]], S = A - B B^H.
    With Psi = L^-H (Schur modes with eigenvalue <= lambda_cutoff_ry2 excluded,
    as the even zero-Ritz policy excludes them, their stored weight reported
    against the same budget), mu = eig(Psi^H G_r Psi) and c = O_r Psi rot. On
    time-reversal-symmetric data B = 0 and the v outputs vanish, so the stored
    model b = sqrt(2) c/mu, poles2 = mu**-2 is the even model. Otherwise it is a
    Galerkin projection with H_r >= 0 and real poles; residues are sign(mu)
    times PSD, negative modes belong to the parent of -q (signed model only).
    |mu| <= keep*max|mu| (poles at infinity) is excluded and its output weight
    is reported. ``keep_budget`` caps the H'_vv keep cut at that many
    directions, the largest first, so K <= keep_budget.

    Returns (b [b,n,R], poles2 [b,R], active [b,R]), the signed model
    (c [b,n,R], mu [b,R], retained [b,R]) and device diagnostics.
    """
    g, h, output, points = pencil
    side, finite = int(g.shape[-1]), int(points.shape[-1])
    half, n_inf = finite // 2, (side - finite) // 2
    if finite % 2 or (side - finite) % 2:
        raise ValueError(f"GATE shared_pole_orientation_pair: got: {finite} finite and {side - finite} infinity columns; want: even counts; why: the ordered cut acts in the paired basis")
    paired = (jnp.all(points[:, half:] == -points[:, :half])
              & jnp.all(active_columns[:, half:finite] == active_columns[:, :half])
              & jnp.all(active_columns[:, finite + n_inf:] == active_columns[:, finite:finite + n_inf]))
    # Traced (a round program): the caller refuses on diagnostics["orientation_paired"].
    if not isinstance(paired, jax.core.Tracer) and not bool(paired):
        raise ValueError(ORIENTATION_PAIR_REFUSAL)
    face = face_sharding(g)
    output_face = face if face_sharding(output) is None else face_sharding(output)
    statics = dict(half=half, finite=finite, n_inf=n_inf)

    live_f = active_columns[:, :half]
    inverse = jnp.where(live_f, 1 / jnp.where(live_f, 2 * points[:, :half], 1), 0)
    # Every large result below stays on the x/y face; eager slicing, concatenation
    # and a + a^H come out replicated ([b,R,R] per rank), which is what ran CrI3 out of memory.
    members = None if face is None else (face,) * 3
    g_ww, g_wv, g_vv = on_face(_paired_member, members, g, inverse, **statics)
    h_ww, h_wv, h_vv = on_face(_paired_member, members, h, inverse, **statics)
    o_w, o_v = on_face(_paired_output, None if output_face is None else (output_face,) * 2,
                       output, inverse, **statics)
    active = jnp.concatenate((live_f, active_columns[:, finite:finite + n_inf]), axis=-1)
    diagonal = jnp.real(jnp.diagonal(h_vv, axis1=-2, axis2=-1))
    diagonal_ok = jnp.all(jnp.where(active, jnp.isfinite(diagonal) & (diagonal > 0), diagonal == 0), axis=-1)
    scale = jnp.where(active, 1 / jnp.sqrt(jnp.where(diagonal > 0, diagonal, 1)), 0)
    sandwich = lambda a: scale[:, :, None] * a * scale[:, None, :]
    g_ww, g_wv, g_vv, h_ww, h_wv, h_vv = (sandwich(a) for a in (g_ww, g_wv, g_vv, h_ww, h_wv, h_vv))
    o_w, o_v = o_w * scale[:, None, :], o_v * scale[:, None, :]
    validity = gates["normalized_gram_validity"]["threshold"]
    gamma, u = eigh(hermitian_part(h_vv))
    largest = gamma[:, -1]
    ratio = gamma[:, 0] / jnp.where(largest > 0, largest, 1)
    keep = ((gamma > gates["normalized_gram_keep"]["threshold"] * largest[:, None]) & (largest[:, None] > 0)
            & _within_budget(gamma, keep_budget))
    count = jnp.sum(keep, axis=-1, dtype=jnp.int64)
    z = u * (keep / jnp.sqrt(jnp.where(keep, gamma, 1)))[:, None, :]
    del u
    metric = matmul(z, matmul(h_vv, z), transa="C")
    null_identity = diagonal_like(~keep, h_vv)
    correction, metric_ok, metric_diagnostics = _metric_inverse_root(
        hermitian_part(metric) + null_identity, matmul=matmul,
        tolerance=gates["retained_subspace_moments"]["threshold"])
    del null_identity
    z = matmul(z, correction) * keep[:, None, :]
    del correction
    metric = matmul(z, matmul(h_vv, z), transa="C")
    wanted_metric = diagonal_like(keep, h_vv)
    metric_relative = (jnp.linalg.norm(metric - wanted_metric, axis=(-2, -1))
                       / jnp.sqrt(jnp.maximum(count, 1)))
    del wanted_metric, h_vv
    # The basis is an explicit operand, so no closure keeps Z alive past its del.
    project = lambda basis, a: matmul(basis, matmul(a, basis), transa="C")
    # Restricted paired pencil on span(Z) in both halves. Its v-block is the metric
    # (identity after correction). On time-reversal-symmetric data H_r = diag(t_s, I),
    # t_s the even route's Z^H H_s Z; once time reversal is broken the halves mix and a
    # w combination can lie in span(v), so a second relative keep cut on H_r removes
    # exactly those redundant combinations before the H-metric Ritz step.
    h_r = on_face(_restricted_block, face, project(z, h_ww), project(z, h_wv), metric)
    del h_ww, h_wv, metric
    g_r = on_face(_restricted_block, face, project(z, g_ww), project(z, g_wv), project(z, g_vv))
    del g_ww, g_wv, g_vv
    o_r = join_columns(matmul(o_w, z), matmul(o_v, z))
    del o_w, o_v, z
    gamma_r, u_r = eigh(h_r)
    top_r = gamma_r[:, -1]
    ratio_r = gamma_r[:, 0] / jnp.where(top_r > 0, top_r, 1)
    keep_r = (gamma_r > gates["normalized_gram_keep"]["threshold"] * top_r[:, None]) & (top_r[:, None] > 0)
    count_r = jnp.sum(keep_r, axis=-1, dtype=jnp.int64)
    y = u_r * (keep_r / jnp.sqrt(jnp.where(keep_r, gamma_r, 1)))[:, None, :]
    del u_r
    null_r = diagonal_like(~keep_r, h_r)
    metric_r = hermitian_part(matmul(y, matmul(h_r, y), transa="C")) + null_r
    # The restricted sources are released before the second metric correction.
    del null_r, h_r
    correction_r, metric_r_ok, paired_metric_diagnostics = _metric_inverse_root(
        metric_r, matmul=matmul, tolerance=gates["retained_subspace_moments"]["threshold"])
    del metric_r
    y = matmul(y, correction_r) * keep_r[:, None, :]
    del correction_r
    mu, rotation = eigh(hermitian_part(matmul(y, matmul(g_r, y), transa="C")))
    del g_r
    c = matmul(o_r, matmul(y, rotation))
    del o_r, y, rotation
    cut = gates["normalized_gram_keep"]["threshold"] * jnp.max(jnp.abs(mu), axis=-1)
    retained = jnp.abs(mu) > cut[:, None]
    positive = retained & (mu > 0)
    weight = jnp.sum(jnp.abs(c) ** 2, axis=-2)
    total = jnp.sum(weight, axis=-1)
    infinite = jnp.sum(jnp.where(retained, 0, weight), axis=-1) / jnp.where(total > 0, total, 1)
    safe = jnp.where(positive, mu, 1)
    b = c * (jnp.sqrt(2.0) / safe * positive)[:, None, :]
    poles2 = jnp.where(positive, 1 / safe**2, 1.0)
    budget = gates["zero_ritz_policy"]["threshold"]["max_dropped_weight_fraction"]
    gram_ok = ((largest > 0) & jnp.all(jnp.isfinite(gamma), axis=-1) & (ratio >= validity)
               & jnp.all(jnp.isfinite(gamma_r), axis=-1) & (ratio_r >= validity))
    diagnostics = {
        **metric_diagnostics,
        "gram_diagonal_positive": diagonal_ok,
        "gram_valid": gram_ok,
        "gram_min_relative": ratio,
        "paired_min_relative": ratio_r,
        "gram_largest": largest,
        "gram_spectrum_finite": jnp.all(jnp.isfinite(gamma), axis=-1),
        "paired_largest": top_r,
        "paired_spectrum_finite": jnp.all(jnp.isfinite(gamma_r), axis=-1),
        "gram_metric_positive": metric_ok,
        "paired_metric_positive": metric_r_ok,
        **{"paired_" + key: value for key, value in paired_metric_diagnostics.items()},
        "paired_rank": count_r,
        "gram_spectrum_relative": gamma / jnp.where(largest > 0, largest, 1)[:, None],
        "retained_rank": count,
        "gram_condition": largest / jnp.min(jnp.where(keep, gamma, jnp.inf), axis=-1),
        "retained_metric_positive": metric_ok & metric_r_ok,
        "retained_metric_relative": metric_relative,
        "positive_count": jnp.sum(positive, axis=-1, dtype=jnp.int64),
        "negative_count": jnp.sum(retained & (mu < 0), axis=-1, dtype=jnp.int64),
        "infinite_weight_fraction": infinite,
        "infinite_weight_ok": infinite <= budget,
        "orientation_paired": jnp.broadcast_to(paired, count.shape),
    }
    return (b, poles2, positive), (c, mu, retained), diagnostics

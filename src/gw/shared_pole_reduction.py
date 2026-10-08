"""Ritz reduction of the shared-pole pencils.

Equilibration, the normalized-Gram keep cut, the coupled Newton-Schulz metric
correction and the Hermitian Ritz step for the even route (W 20 of
docs/theory/shared-pole-w-model.md), and the paired basis, its second cut and
the signed model for the ordered route (W 27).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from distrib_la import (diagonal_like, face_sharding, hermitian_part,
                        join_columns, on_face)
from gw.shared_pole_pencil import _adjoint, _matrix_layout


def _metric_inverse_root(metric, *, matmul, tolerance, matrix_sharding=None):
    """Correct a dimensionless Hermitian metric by coupled Newton–Schulz.

    ``metric`` is [b,R,R], complex128 in the caller's face layout. Products
    use the resolved distrib_la GEMM. Y starts at A and Z at I; the coupled
    update T=(3I-ZY)/2, Y=YT, Z=TZ computes A**(-1/2) without eigenvectors.
    The initial infinity norm bounds the spectral error. For radius d<1,
    d_next <= d**2*(3+d)/4 <= d**2; choose the entire iteration count from
    that initial bound, never from an on-device residual convergence test.
    The returned diagnostics include the measured ZAZ-I residual and guard.
    """
    identity = _matrix_layout(diagonal_like(jnp.ones(metric.shape[:1] + metric.shape[-1:]), metric), matrix_sharding)
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



_UNIT_ROUNDOFF = float(jnp.finfo(jnp.float64).eps) / 2


def _frobenius_by_rows(entry, rows, block=512):
    """sqrt(sum of entry**2) over all rows, one bounded [b, block, R] row block at a time."""
    starts = jnp.arange(-(-int(rows) // block)) * block

    def one(start):
        idx = start + jnp.arange(block)
        valid = idx < rows
        value = entry(jnp.minimum(idx, rows - 1))
        return jnp.sum(jnp.where(valid[None, :, None], value, 0.0) ** 2, axis=(-2, -1))
    return jnp.sqrt(jnp.sum(jax.lax.map(one, starts), axis=0))


def _divided_difference_scale(x, xr, o, orow, d, drow):
    """Uncancelled magnitude of one resolvent-identity Gram entry (W 18).

    G_ab = (Q_a^H O_b - O_a^H Q_b)/(x_b - conj x_a) with unit directions: the
    numerator's two inner products have magnitudes <= |O_a|, |O_b|, so its
    rounding is gamma_n (|O_a| + |O_b|) and the entry's is that over the
    denominator. A confluent entry is the derivative action, <= |D|.
    Rows [b, blk] (xr, orow, drow) against columns [b, R] (x, o, d).
    """
    den = jnp.abs(x[:, None, :] - jnp.conj(xr)[:, :, None])
    mag = jnp.maximum(1.0, jnp.maximum(jnp.abs(xr)[:, :, None], jnp.abs(x)[:, None, :]))
    conf = den <= 8 * jnp.finfo(jnp.float64).eps * mag
    return jnp.where(conf, 0.5 * (drow[:, :, None] + d[:, None, :]),
                     (orow[:, :, None] + o[:, None, :]) / jnp.where(conf, 1.0, den))


def gram_rounding_floor(scale, rounding, output_norm, *, finite):
    """Rounding floor of the equilibrated even Gram's spectrum (Weyl bound).

    The exact Gram X^H X is PSD, so a computed eigenvalue below zero is
    rounding. Each entry is a divided difference of sample actions (W 18,
    W 19) whose float64 error is at most gamma_n times its uncancelled
    magnitude Sigma_ab (``_divided_difference_scale``; infinity rows
    Q_inf^H O_b carry |O_b|, the moment block |2 M1 Q_inf|), with
    gamma_n = n u / (1 - n u) for inner products of length n. Equilibration
    scales Sigma by D^-1/2 on both sides, and Weyl's inequality gives
    gamma_min >= -||E||_2 >= -gamma_n ||D^-1/2 Sigma D^-1/2||_F. The
    eigensolver adds R u gamma_max. The floor is therefore relative to the
    cancelled scale of the entries, not to the Gram itself: close supports,
    small diagonals and large outputs raise it; a well-separated pencil
    gets a floor far below the old fixed 1e-7.

    ``scale`` [b,R] is 1/sqrt(diag G) (0 on inert columns); ``rounding`` holds
    ``points`` [b,F] (s, Ry^2), ``derivative_norm`` [b,F] and ``rows`` (n);
    ``output_norm`` [b,R] the column norms of O (infinity columns 2 M1 Q_inf).
    Returns the absolute floor [b] on the equilibrated spectrum, before R u gamma_max.
    """
    side = scale.shape[-1]
    n = float(rounding["rows"])
    gamma_n = n * _UNIT_ROUNDOFF / (1 - n * _UNIT_ROUNDOFF)
    pad = ((0, 0), (0, side - finite))
    x = jnp.pad(jnp.asarray(rounding["points"], jnp.complex128), pad)
    d = jnp.pad(jnp.asarray(rounding["derivative_norm"], jnp.float64), pad)
    o = output_norm
    inf = jnp.arange(side) >= finite

    def entry(idx):
        ri, ci = inf[idx][None, :, None], inf[None, None, :]
        orow = o[:, idx]
        finite_block = _divided_difference_scale(x, x[:, idx], o, orow, d, d[:, idx])
        mixed = jnp.where(ri, o[:, None, :], orow[:, :, None])
        moments = 0.5 * (orow[:, :, None] + o[:, None, :])
        value = jnp.where(ri & ci, moments, jnp.where(ri | ci, mixed, finite_block))
        return scale[:, idx][:, :, None] * value * scale[:, None, :]
    return gamma_n * _frobenius_by_rows(entry, side)


def _within_budget(gamma, budget):
    """The largest ``budget`` entries of each ascending spectrum row (all when None),
    closed downward over the edge multiplet: an entry tied to its neighbour below
    the cut (the direction selection's relative multiplet tolerance, or within
    the eigensolver's R u gamma_max) leaves with it.  The kept span then never
    depends on the eigenbasis a solver route chose inside a degenerate multiplet,
    and K <= budget, so every carrier sized on the budget still holds it."""
    side = gamma.shape[-1]
    if budget is None or int(budget) >= side:
        return True
    from gw.shared_pole_recipe import shared_real_pole_v1_r3b
    edge = side - int(budget)
    lo, hi = gamma[:, edge - 1:-1], gamma[:, edge:]
    tied = jnp.abs(hi - lo) <= jnp.maximum(
        shared_real_pole_v1_r3b["multiplet_relative_tolerance"] * jnp.maximum(jnp.abs(hi), jnp.abs(lo)),
        side * _UNIT_ROUNDOFF * jnp.abs(gamma[:, -1:]))
    dropped = jnp.sum(jnp.cumprod(tied, axis=-1), axis=-1)
    # One tied run over the whole kept set has no multiplet edge: the index cut stays.
    dropped = jnp.where(dropped < int(budget), dropped, 0)
    return jnp.arange(side)[None, :] >= edge + dropped[:, None]


def _validity_floor(scale, rounding, output_norm, largest, *, gates):
    """Absolute floor [b] on the equilibrated Gram's smallest eigenvalue.

    Every route has the fixed relative row ``normalized_gram_validity``. With
    the support geometry (``rounding``, the even route) the floor is the larger
    of that and the propagated float64 bound of ``gram_rounding_floor`` plus
    the eigensolver's R u gamma_max (gate row ``gram_rounding_validity``).
    The bound covers forming G from the samples, not the samples' own error:
    it read 8e-13 against a -8.7e-10 computed minimum on the hsuite Na SC
    stage, which the fixed row admits.
    """
    fixed = -gates["normalized_gram_validity"]["threshold"] * jnp.maximum(largest, 0)
    if rounding is None:
        return fixed
    bound = gram_rounding_floor(scale, rounding, output_norm, finite=int(rounding["points"].shape[-1]))
    factor = float(gates["gram_rounding_validity"]["threshold"]["bound_factor"])
    # The larger of the fixed floor and the computed float64 bound; sample error is not bounded (future work).
    return jnp.maximum(fixed, factor * bound + scale.shape[-1] * _UNIT_ROUNDOFF * jnp.maximum(largest, 0))


def reduce_shared_pole_pencil(pencil, active_columns, *, eigh, matmul, gates, keep_budget=None,
                              rounding=None):
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
    rounding : mapping, optional
        ``points`` [b,F] (s), ``derivative_norm`` [b,F] and ``rows`` (n) of the
        finite columns, for the Gram validity floor (``gram_rounding_floor``).
        Without it (direct unit calls) the floor is the legacy fixed row.

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
    # Raw column norms, before equilibration: the rounding floor's |O_a|.
    output_norm = None if rounding is None else jnp.linalg.norm(output, axis=-2)
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
    floor = _validity_floor(scale, rounding, output_norm, largest, gates=gates)
    gram_ok = ((largest > 0) & jnp.all(jnp.isfinite(gamma), axis=-1)
               & (gamma[:, 0] >= -floor))
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
        "gram_floor_relative": floor / jnp.where(largest > 0, largest, 1),
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


def _schur_basis(y_s, bh_y_s, kept):
    """[[Y_S, 0], [-B^H Y_S, diag(kept)]] [b, 2R, 2R]: P diag(Y_S, I_kept) with P = [[I, 0], [-B^H, I]]."""
    eye = jnp.eye(y_s.shape[-1], dtype=y_s.dtype)[None] * kept[:, None, :].astype(y_s.dtype)
    return jnp.concatenate((jnp.concatenate((y_s, jnp.zeros_like(y_s)), axis=-1),
                            jnp.concatenate((-bh_y_s, eye), axis=-1)), axis=-2)


def _restricted_block(ww, wv, vv):
    """Hermitian [[ww, wv], [wv^H, vv]] of the restricted paired pencil."""
    return hermitian_part(jnp.concatenate(
        (jnp.concatenate((ww, wv), axis=-1), jnp.concatenate((_adjoint(wv), vv), axis=-1)), axis=-2))


def reduce_ordered_shared_pole_pencil(pencil, active_columns, *, eigh, matmul, gates, keep_budget=None,
                                     retain_span=False, matrix_sharding=None, gram_keep=None, carrier=None):
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
    directions, the largest first, so K <= keep_budget. On a rank-local
    pencil the kept span is solved on its last ``carrier`` columns (default
    ``keep_budget``); a kept count above the carrier (``retained_rank``) is
    the caller's to rerun wider. A face pencil does the same on a ``carrier``
    of at least ``keep_budget`` that tiles the mesh. The second cut acts on
    the eigenvalues of S relative to max(top(S), 1), the top of diag(S, I)
    (the kept v directions stay), and Y = L^-H on the kept
    span is P diag(U_S Gamma_S^-1/2, I) with P = [[I, 0], [-B^H, I]], corrected
    by Newton–Schulz against the computed H_r.

    Returns (b [b,n,R], poles2 [b,R], active [b,R]), the signed model
    (c [b,n,R], mu [b,R], retained [b,R]) and device diagnostics.
    With ``retain_span``, also return the original-pencil coefficient map
    Y [b,R,K] on the solve's own K columns (K twice the Ritz carrier, R on
    a face pencil given none), with O Y = c[..., :K] and
    Y.H H Y = diag(retained[..., :K]), for the CT joint projection. The
    signed model's columns past K are zero padding, so Y carries none.
    This map stays inside the parent-local round.
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
    stage = paired_members(pencil, active_columns, gates=gates, matrix_sharding=matrix_sharding)
    gamma, u = eigh(hermitian_part(stage["h_vv"]))
    stage = keep_stage(stage, gamma, u, matmul=matmul, gates=gates, keep_budget=keep_budget,
                       retain_span=retain_span, matrix_sharding=matrix_sharding, gram_keep=gram_keep,
                       carrier=carrier)
    gamma_r, u_r = eigh(stage["schur"])
    stage = paired_stage(stage, gamma_r, u_r, matmul=matmul, gates=gates, matrix_sharding=matrix_sharding,
                         gram_keep=gram_keep)
    mu, rotation = eigh(stage["reduced"])
    return output_stage(stage, mu, rotation, matmul=matmul, gates=gates, retain_span=retain_span,
                        matrix_sharding=matrix_sharding)


# The four GEMM stages of the paired reduction, with the three eighs between them.
# ``reduce_ordered_shared_pole_pencil`` composes them inside one program (the local
# round, the face round); the staged face route runs each stage over a batch of
# parents and each eigh over the round's stack (``shared_pole_execution``).
# A stage takes and returns a dict of arrays; no array crosses a stage boundary
# that the next stage does not read.

def paired_members(pencil, active_columns, *, gates, matrix_sharding=None):
    """Stage 1: the paired-basis members of (G, H, O), equilibrated; ``h_vv`` goes to the first eigh."""
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
    g_ww, g_wv, g_vv, h_ww, h_wv, h_vv, o_w, o_v = (
        _matrix_layout(a, matrix_sharding) for a in
        (g_ww, g_wv, g_vv, h_ww, h_wv, h_vv, o_w, o_v))
    active = jnp.concatenate((live_f, active_columns[:, finite:finite + n_inf]), axis=-1)
    diagonal = jnp.real(jnp.diagonal(h_vv, axis1=-2, axis2=-1))
    diagonal_ok = jnp.all(jnp.where(active, jnp.isfinite(diagonal) & (diagonal > 0), diagonal == 0), axis=-1)
    scale = jnp.where(active, 1 / jnp.sqrt(jnp.where(diagonal > 0, diagonal, 1)), 0)
    sandwich = lambda a: scale[:, :, None] * a * scale[:, None, :]
    g_ww, g_wv, g_vv, h_ww, h_wv, h_vv = (sandwich(a) for a in (g_ww, g_wv, g_vv, h_ww, h_wv, h_vv))
    o_w, o_v = o_w * scale[:, None, :], o_v * scale[:, None, :]
    return dict(g_ww=g_ww, g_wv=g_wv, g_vv=g_vv, h_ww=h_ww, h_wv=h_wv, h_vv=h_vv, o_w=o_w, o_v=o_v,
                scale=scale, inverse=inverse, diagonal_ok=diagonal_ok, paired=paired, side=side, half=half)


def keep_stage(stage, gamma, u, *, matmul, gates, keep_budget=None, retain_span=False,
               matrix_sharding=None, gram_keep=None, carrier=None):
    """Stage 2: the H'_vv keep cut, its metric correction and the restricted pencil; ``schur`` goes to the second eigh."""
    scale = stage["scale"]
    h_vv, h_ww, h_wv = stage["h_vv"], stage["h_ww"], stage["h_wv"]
    face = None
    validity = gates["normalized_gram_validity"]["threshold"]
    keep_cut = gates["normalized_gram_keep"]["threshold"] if gram_keep is None else gram_keep
    largest = gamma[:, -1]
    ratio = gamma[:, 0] / jnp.where(largest > 0, largest, 1)
    floor = _validity_floor(scale, None, None, largest, gates=gates)
    keep = ((gamma > keep_cut * largest[:, None]) & (largest[:, None] > 0)
            & _within_budget(gamma, keep_budget))
    count = jnp.sum(keep, axis=-1, dtype=jnp.int64)
    # Budget-excluded columns are exactly zero; omit them from the dense work
    # below. A rank-local pencil keeps ``carrier`` columns (default
    # ``keep_budget``); a face pencil keeps them only when its caller gives a
    # carrier that tiles the mesh (``shared_pole_execution.face_ritz_carrier``).
    width = gamma.shape[-1]
    if (keep_budget is not None and face is None
            and (matrix_sharding is None or carrier is not None)):
        width = min(width, max(1, int(keep_budget if carrier is None else carrier)))
    kept, values = keep[:, -width:], gamma[:, -width:]
    z = _matrix_layout(u[..., -width:] * (kept / jnp.sqrt(jnp.where(kept, values, 1)))[:, None, :],
                       matrix_sharding)
    del u
    metric = matmul(z, matmul(h_vv, z), transa="C")
    null_identity = diagonal_like(~kept, metric)
    correction, metric_ok, metric_diagnostics = _metric_inverse_root(
        hermitian_part(metric) + null_identity, matmul=matmul,
        tolerance=gates["retained_subspace_moments"]["threshold"], matrix_sharding=matrix_sharding)
    del null_identity
    z = matmul(z, correction) * kept[:, None, :]
    del correction
    metric = matmul(z, matmul(h_vv, z), transa="C")
    wanted_metric = diagonal_like(kept, metric)
    metric_relative = (jnp.linalg.norm(metric - wanted_metric, axis=(-2, -1))
                       / jnp.sqrt(jnp.maximum(count, 1)))
    del wanted_metric, h_vv
    # The basis is an explicit operand, so no closure keeps Z alive past its del.
    project = lambda basis, a: matmul(basis, matmul(a, basis), transa="C")
    # Restricted paired pencil on span(Z) in both halves. Its v-block is the metric
    # (identity after correction). On time-reversal-symmetric data H_r = diag(t_s, I),
    # t_s the even route's Z^H H_s Z; once time reversal is broken the halves mix and a
    # w combination can lie in span(v), so a second relative keep cut removes exactly
    # those redundant combinations before the H-metric Ritz step. The cut acts on
    # the Schur complement S = A - B B^H of the unit v-block: P^H H_r P = diag(S, I) with
    # P = [[I, 0], [-B^H, I]], so only S is eigendecomposed and the structural unit
    # half of H_r never enters the eigensolver (whole-H_r eigh: backward error
    # 8e-4 top on Fe 4^3 complete-basis parents, claim of the PAIREDHR lane).
    a_r, b_r = project(z, h_ww), project(z, h_wv)
    h_r = _matrix_layout(on_face(_restricted_block, face, a_r, b_r, metric), matrix_sharding)
    schur = _matrix_layout(hermitian_part(a_r - matmul(b_r, b_r, transb="C")), matrix_sharding)
    del h_ww, h_wv, metric, a_r
    g_r = _matrix_layout(on_face(_restricted_block, face, project(z, stage["g_ww"]), project(z, stage["g_wv"]),
                                 project(z, stage["g_vv"])), matrix_sharding)
    o_r = _matrix_layout(join_columns(matmul(stage["o_w"], z), matmul(stage["o_v"], z)), matrix_sharding)
    out = dict(schur=schur, h_r=h_r, g_r=g_r, o_r=o_r, b_r=b_r, kept=kept, keep=keep, gamma=gamma, largest=largest,
               ratio=ratio, floor=floor, count=count, width=width, metric_ok=metric_ok,
               metric_relative=metric_relative, metric_diagnostics=metric_diagnostics,
               scale=scale, inverse=stage["inverse"], diagonal_ok=stage["diagonal_ok"],
               paired=stage["paired"], side=stage["side"], half=stage["half"])
    if retain_span:
        out["paired_span"] = scale[:, :, None] * z
    return out


def paired_stage(stage, gamma_r, u_r, *, matmul, gates, matrix_sharding=None, gram_keep=None):
    """Stage 3: the Schur cut, Y = L^-H on the kept span and its metric correction; ``reduced`` goes to the last eigh."""
    face = None
    # The Schur cut uses the same relative keep as the H'_vv cut (the sector threshold).
    keep_cut = gates["normalized_gram_keep"]["threshold"] if gram_keep is None else gram_keep
    kept, b_r, h_r, g_r = stage["kept"], stage["b_r"], stage["h_r"], stage["g_r"]
    # diag(S, I) is H_r's congruent form: its spectrum is spec(S) plus the unit
    # block, so the relative cut and the validity ratio are taken against
    # max(top(S), 1), as the whole-H_r cut was against top(H_r) >= 1. Relative
    # to top(S) alone, a parent whose w directions all lie in span(v) (S at
    # round-off) would keep round-off modes and fail the Gram validity gate.
    top_r = jnp.maximum(gamma_r[:, -1], 1.0)
    ratio_r = gamma_r[:, 0] / top_r
    keep_s = gamma_r > keep_cut * top_r[:, None]
    y_s = u_r * (keep_s / jnp.sqrt(jnp.where(keep_s, gamma_r, 1)))[:, None, :]
    del u_r
    # Y = P diag(Y_S, I_kept): Schur modes, then the kept v directions.
    y = _matrix_layout(on_face(_schur_basis, face, y_s, matmul(b_r, y_s, transa="C"), kept), matrix_sharding)
    del y_s, b_r
    keep_r = jnp.concatenate((keep_s, kept), axis=-1)
    count_r = jnp.sum(keep_r, axis=-1, dtype=jnp.int64)
    null_r = diagonal_like(~keep_r, h_r)
    metric_r = hermitian_part(matmul(y, matmul(h_r, y), transa="C")) + null_r
    # The restricted sources are released before the second metric correction.
    del null_r, h_r
    correction_r, metric_r_ok, paired_metric_diagnostics = _metric_inverse_root(
        metric_r, matmul=matmul, tolerance=gates["retained_subspace_moments"]["threshold"], matrix_sharding=matrix_sharding)
    del metric_r
    y = matmul(y, correction_r) * keep_r[:, None, :]
    del correction_r
    reduced = hermitian_part(matmul(y, matmul(g_r, y), transa="C"))
    out = {k: v for k, v in stage.items() if k not in ("schur", "h_r", "b_r", "g_r")}
    out.update(reduced=reduced, y=y, keep_r=keep_r, count_r=count_r, gamma_r=gamma_r, ratio_r=ratio_r,
               metric_r_ok=metric_r_ok,
               paired_metric_residual_relative=paired_metric_diagnostics["metric_inverse_root_residual_relative"],
               paired_metric_iterations=paired_metric_diagnostics["metric_inverse_root_iterations"])
    return out


def output_stage(stage, mu, rotation, *, matmul, gates, retain_span=False, matrix_sharding=None):
    """Stage 4: the Ritz step's outputs, the signed and positive models and the diagnostics."""
    y, o_r, kept, keep_r = stage["y"], stage["o_r"], stage["kept"], stage["keep_r"]
    inverse, scale, side, half, width = stage["inverse"], stage["scale"], stage["side"], stage["half"], stage["width"]
    gamma, largest, ratio, floor, count = stage["gamma"], stage["largest"], stage["ratio"], stage["floor"], stage["count"]
    gamma_r, ratio_r, count_r = stage["gamma_r"], stage["ratio_r"], stage["count_r"]
    validity = gates["normalized_gram_validity"]["threshold"]
    c = matmul(o_r, matmul(y, rotation))
    if retain_span:
        paired_span = stage["paired_span"]
        ritz = matmul(y, rotation)
        span_w = matmul(paired_span, ritz[:, :width])
        span_v = matmul(paired_span, ritz[:, width:])
        # Undo w=(X(z)+X(-z))/2, v=(X(z)-X(-z))/(2z),
        # w_inf=k1 and v_inf=k0 (report equation 5.6).
        coefficients = jnp.concatenate((
            .5 * span_w[:, :half] + inverse[:, :, None] * span_v[:, :half],
            .5 * span_w[:, :half] - inverse[:, :, None] * span_v[:, :half],
            span_v[:, half:], span_w[:, half:]), axis=-2)
        coefficients = _matrix_layout(coefficients, matrix_sharding)
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
    gram_ok = ((largest > 0) & jnp.all(jnp.isfinite(gamma), axis=-1) & (gamma[:, 0] >= -floor)
               & jnp.all(jnp.isfinite(gamma_r), axis=-1) & (ratio_r >= validity))
    diagnostics = {
        **stage["metric_diagnostics"],
        "paired_metric_inverse_root_residual_relative": stage["paired_metric_residual_relative"],
        "paired_metric_inverse_root_iterations": stage["paired_metric_iterations"],
        "gram_diagonal_positive": stage["diagonal_ok"],
        "gram_valid": gram_ok,
        "gram_min_relative": ratio,
        "gram_floor_relative": floor / jnp.where(largest > 0, largest, 1),
        "paired_min_relative": ratio_r,
        "paired_rank": count_r,
        "gram_spectrum_relative": gamma / jnp.where(largest > 0, largest, 1)[:, None],
        "retained_rank": count,
        "gram_condition": largest / jnp.min(jnp.where(stage["keep"], gamma, jnp.inf), axis=-1),
        "retained_metric_positive": stage["metric_ok"] & stage["metric_r_ok"],
        "retained_metric_relative": stage["metric_relative"],
        "positive_count": jnp.sum(positive, axis=-1, dtype=jnp.int64),
        "negative_count": jnp.sum(retained & (mu < 0), axis=-1, dtype=jnp.int64),
        "infinite_weight_fraction": infinite,
        "infinite_weight_ok": infinite <= budget,
        "orientation_paired": jnp.broadcast_to(stage["paired"], count.shape),
    }
    if retain_span:
        coefficients = coefficients * retained[:, None, :]
    pad = side - mu.shape[-1]
    b, c = (jnp.pad(a, ((0, 0), (0, 0), (0, pad))) for a in (b, c))
    poles2 = jnp.pad(poles2, ((0, 0), (0, pad)), constant_values=1)
    mu, positive, retained = (jnp.pad(a, ((0, 0), (0, pad))) for a in (mu, positive, retained))
    result = (b, poles2, positive), (c, mu, retained), diagnostics
    if retain_span:
        return (*result, coefficients)
    return result


def real_galerkin_pencil(pencil, mixed, coefficients, original_active, *, matmul, matrix_sharding=None):
    """Phase-balanced real input columns from an authenticated Gamma Ritz span.

    Coefficients contain ONLY final active columns of the first compression,
    in matching model order. G/H/K/L retain the original packed input order.
    Normalizing input columns before these contractions avoids amplifying
    their dynamic range. This changes the Galerkin space, not a model field.
    """
    from gw.shared_pole_pencil import _matrix_concat
    g, h, output = pencil
    k, l = mixed
    diagonal = jnp.real(jnp.diagonal(g, axis1=-2, axis2=-1))
    scale = jnp.where(original_active, 1 / jnp.sqrt(jnp.where(diagonal > 0, diagonal, 1)), 0)
    ye = _matrix_layout(coefficients / jnp.where(scale > 0, scale, 1)[:, :, None], matrix_sharding)
    normalize = lambda a: _matrix_layout(scale[:, :, None] * a * scale[:, None, :], matrix_sharding)
    m = matmul(ye, matmul(normalize(g), ye), transa="C")
    s = matmul(ye, matmul(normalize(k), ye), transa="T")
    a = matmul(ye, matmul(normalize(h), ye), transa="C")
    lr = matmul(ye, matmul(normalize(l), ye), transa="T")
    b = matmul(output, coefficients)
    sd = jnp.diagonal(s, axis1=-2, axis2=-1)
    phase = jnp.where(jnp.abs(sd) > 0, jnp.exp(.5j * (jnp.pi / 2 - jnp.angle(sd))), 1 + 0j)
    hermitian_gauge = lambda x: _matrix_layout(jnp.conj(phase)[:, :, None] * x * phase[:, None, :], matrix_sharding)
    ordinary_gauge = lambda x: _matrix_layout(phase[:, :, None] * x * phase[:, None, :], matrix_sharding)
    m, a, s, lr = hermitian_gauge(m), hermitian_gauge(a), ordinary_gauge(s), ordinary_gauge(lr)
    b = _matrix_layout(b * phase[:, None, :], matrix_sharding)
    def augment(h_, k_):
        rr, ri = .5 * jnp.real(h_ + k_), .5 * jnp.imag(k_ + h_)
        ir, ii = .5 * jnp.imag(k_ - h_), .5 * jnp.real(h_ - k_)
        return _matrix_concat((_matrix_concat((rr, ri), -1, matrix_sharding),
                               _matrix_concat((ir, ii), -1, matrix_sharding)), -2, matrix_sharding)
    return augment(m, s), augment(a, lr), _matrix_concat((jnp.real(b), jnp.imag(b)), -1, matrix_sharding)


def reduce_real_galerkin_pencil(pencil, original_qi, *, eigh, matmul, gates, keep_budget, matrix_sharding=None, active_columns=None):
    """Reduce real Gamma columns and check the original physical infinity anchors.

    Original x_inf=B0.T Q_inf; with real O=B0 X_real, their cross Gram is
    O.T Q_inf. E=Y Y.H (O.T Q_inf) represents their projection into the NEW
    final Ritz span. The source moment owner checks that fresh projected
    anchor. This does not reuse the old input selector or an identity test.
    Physical full-bank moments/passivity/held checks remain caller duties.
    """
    from gw.shared_pole_gates import apply_shared_pole_zero_policy, retained_moment_identity
    g, h, output = pencil
    if any(a.dtype != jnp.float64 for a in pencil):
        raise TypeError("real Galerkin reduction requires float64 input columns")
    active = jnp.ones(g.shape[:1] + g.shape[-1:], dtype=bool) if active_columns is None else active_columns
    model, reduction, y = reduce_shared_pole_pencil(pencil, active, eigh=eigh, matmul=matmul,
        gates=gates, keep_budget=keep_budget)
    model, zero = apply_shared_pole_zero_policy(model, gates=gates)
    y = _matrix_layout(y * model[2][:, None, :], matrix_sharding)
    # The physical-anchor consumer has complex Q_inf even when its response
    # and the preceding eigensolves are real. Cast at this boundary only.
    gc, hc, oc, yc = (_matrix_layout(a.astype(jnp.complex128), matrix_sharding) for a in (g, h, output, y))
    cross = matmul(oc, original_qi, transa="T")
    projection = matmul(yc, matmul(yc, cross, transa="C"))
    physical_model = (model[0].astype(jnp.complex128), model[1], model[2])
    retained = retained_moment_identity((gc, hc, oc), yc, physical_model, projection, matmul=matmul)
    return physical_model, (reduction, zero, retained), yc

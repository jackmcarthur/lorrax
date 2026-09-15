"""Tangential Hermite/Ritz construction of physical shared real-pole W.

The physical convention is W(s) = b (s - Lambda)^-1 b.H, s = z_Ry**2;
S_m = 2 M_(2m+1).  Thus b has units Ry**(3/2), Lambda Ry**2, and
M1/M3 are physical moments in Ry**3/Ry**5, never bare chi coefficients.

Sample matrices are consumed in bounded batches.  Only narrow direction,
output and derivative-action panels survive to the pencil assembly.  Dense
products and eigensolves are supplied by the resolved distrib_la operations;
there is no local vendor or alternative eigensolver in this physics owner.
"""

from __future__ import annotations

from functools import lru_cache, partial

import jax
import jax.numpy as jnp
from common import timing


def shared_pole_byte_terms(meta, *, mesh_xy, resolution, pencil_side,
                           parent_batch, sample_batch, phase="reduction"):
    """Price constructor carriers; the map CapacityLedger owns admission.

    Selection holds samples, current narrow actions and the n/2n direction
    solve; it has no R-by-R pencil. Reduction holds the actual selected
    pencil. Model checks hold factors and bounded samples, with no pencil.
    Native workspace is separately supplied by the service. No threshold
    or independent capacity policy lives in this constructor helper.
    """
    import math

    p = int(mesh_xy.shape["x"]) * int(mesh_xy.shape["y"])
    # Constructor carriers are mu x mu charge operators on every admitted deck.
    packed = int(meta.n_rmu_padded)
    b, a, r = int(parent_batch), int(sample_batch), int(pencil_side)
    if min(packed, b, a) <= 0 or r < 0:
        raise ValueError("GATE shared_pole_capacity: got: invalid extents; want: positive basis/batches and nonnegative pencil; why: live-set pricing")
    dense_copies = math.ceil(b / p) if resolution.layout == "local" else b / p
    if phase == "selection":
        dense = 24 * packed**2
        sample_faces = max(2*a, 2)
    elif phase == "reduction":
        dense = 14 * r*r + 12 * packed * r
        sample_faces = 0
    elif phase == "model":
        dense = 8 * packed**2 + 4 * packed * r
        sample_faces = max(2*a, 2)
    else:
        raise ValueError(f"unknown shared-pole capacity phase: {phase}")
    terms = {
        "sample_or_moment_batch": math.ceil(16*b*sample_faces*packed**2/p),
        "narrow_actions": math.ceil(16*b*3*packed*r/p),
        "replicated_scalars": 8*b*(12*r+4*packed),
        "phase_dense_temporaries": math.ceil(16*dense_copies*dense),
    }
    return {"terms_bytes_per_rank": terms,
            "resident_bytes_per_rank": sum(terms.values()),
            "layout": resolution.layout, "phase": phase, "pencil_side": r,
            "parent_batch": b, "sample_batch": a}


def _adjoint(a):
    return jnp.conj(jnp.swapaxes(a, -1, -2))


def _hermitian(a):
    return (a + _adjoint(a)) * 0.5


def _diagonal_face(values, matrix):
    """Materialize a diagonal directly in the supplied pencil face layout."""
    import jax
    def build(d):
        return jnp.eye(d.shape[-1], dtype=matrix.dtype)[None] * d[:, None, :]
    if isinstance(matrix, jax.core.Tracer):
        return build(values)
    return jax.jit(build, out_shardings=matrix.sharding)(values)


def finite_pencil_column(left, right, *, matmul):
    """Form one block column of the resolvent-identity pencil.

    Parameters
    ----------
    left : tuple
        ``(s, Q, O)`` with s [R] or [b,R] in Ry**2, Q/O [b,n,R] complex128
        face tiles and O_a = W(s_a) Q_a. Repeated entries of s label
        separate tangential columns, including conjugate and role states.
    right : tuple
        ``(s_b, Q_b, O_b, D_b)`` with panels [b,n,r] in the same layout,
        s_b scalar or [b,r], O_b = W(s_b) Q_b and D_b = dW(s_b)/ds Q_b.
    matmul : callable
        Resolved service GEMM; accepts ``transa='C'`` for the adjoint.

    Returns
    -------
    g, h : arrays
        [b,R,r] complex128 face tiles. They are X.H X and X.H T X
        blocks for X_b = (s_b-T)^-1 b.H Q_b (algorithm guide, section 4).
    """
    sa, qa, oa = left
    sb, qb, ob, db = right
    a = matmul(oa, qb, transa="C")
    # Production assembles the complete square finite block with shared
    # panels. For distinct left/right blocks the second product is needed.
    b = _adjoint(a) if qa is qb and oa is ob else matmul(qa, ob, transa="C")
    derivative = matmul(qa, db, transa="C")
    sa = jnp.broadcast_to(sa, (qa.shape[0], qa.shape[-1]))
    sb = jnp.broadcast_to(sb, (qb.shape[0], qb.shape[-1]))
    denominator = sb[:, None, :] - jnp.conj(sa[:, :, None])
    # This is the inherited floating-point equality test for confluent s,
    # not a physical support-merging tolerance. Roles are never merged.
    scale = jnp.maximum(1.0, jnp.maximum(jnp.abs(sa[:, :, None]),
                                        jnp.abs(sb[:, None, :])))
    confluent = jnp.abs(denominator) <= 8 * jnp.finfo(jnp.float64).eps * scale
    safe = jnp.where(confluent, 1.0 + 0j, denominator)
    g = jnp.where(confluent, -derivative, (a - b) / safe)
    return g, sb[:, None, :] * g - a


def infinity_pencil_column(finite, infinity, *, matmul):
    """Form infinity rows using physical M1 and M3 (guide section 4).

    ``finite=(s,Q,O)`` carries [R] or [b,R], [b,n,R], [b,n,R].
    ``infinity=(Q_inf,M1_Q_inf,M3_Q_inf)`` carries three [b,n,r_inf]
    complex128 face panels. Returns G_inf,finite, H_inf,finite,
    G_inf,inf, H_inf,inf and O_inf. No full moment matrix is retained.
    """
    s, q, output = finite
    qi, m1qi, m3qi = infinity
    gi = matmul(qi, output, transa="C")
    s = jnp.broadcast_to(s, (q.shape[0], q.shape[-1]))
    hi = gi * s[:, None, :] - 2 * matmul(m1qi, q, transa="C")
    gii = 2 * matmul(qi, m1qi, transa="C")
    hii = 2 * matmul(qi, m3qi, transa="C")
    return gi, hi, gii, hii, 2 * m1qi


def assemble_shared_pole_pencil(states, infinity, *, matmul):
    """Assemble G, H and O from bounded-sample action panels.

    ``states`` is an ordered sequence of ``(s,Q,WQ,dWQ)`` tuples, where
    s is scalar or [b,r_a] Ry**2; panels are complex128 [b,n,r_a] face tiles.
    Each state preserves its own role, even when samples are shared.
    ``infinity`` contains Q_inf, M1 Q_inf, M3 Q_inf [b,n,r_inf].
    Only narrow panels and the dense pencil are resident here; full W
    samples must already have been released by their bounded producer.
    """
    q = jnp.concatenate([state[1] for state in states], axis=-1)
    output = jnp.concatenate([state[2] for state in states], axis=-1)
    derivative = jnp.concatenate([state[3] for state in states], axis=-1)
    s = jnp.concatenate([jnp.broadcast_to(jnp.asarray(state[0], jnp.complex128),
                         (state[1].shape[0], state[1].shape[-1]))
                         for state in states], axis=-1)
    finite = (s, q, output)
    # All finite columns share three products. The confluent derivative and
    # support coordinates remain column-specific, including repeated roles.
    g, h = finite_pencil_column(finite, (s, q, output, derivative), matmul=matmul)
    gi, hi, gii, hii, oi = infinity_pencil_column(finite, infinity, matmul=matmul)
    g = jnp.concatenate((jnp.concatenate((g, _adjoint(gi)), axis=-1),
                         jnp.concatenate((gi, gii), axis=-1)), axis=-2)
    h = jnp.concatenate((jnp.concatenate((h, _adjoint(hi)), axis=-1),
                         jnp.concatenate((hi, hii), axis=-1)), axis=-2)
    return _hermitian(g), _hermitian(h), jnp.concatenate((output, oi), axis=-1)


def ordered_infinity_pencil_column(finite, infinity, *, matmul):
    """Infinity rows of the linear particle-hole pencil (z s3 - M).

    ``finite=(z,Q,O)`` carries complex z in Ry ([R] or [b,R]) and [b,n,R]
    face panels. ``infinity=(Q_inf, M0 Q_inf, M1 Q_inf, M2 Q_inf, M3 Q_inf)``
    with physical z-moments Wc(z) = sum_n 2 M_n z^-(n+1); M0 and M2 are odd
    under time reversal. States are k0 = s3 C^H Q and k1 = s3 M s3 C^H Q.
    Returns G/H infinity-finite rows [b,2r,R], the [b,2r,2r] infinity
    blocks and the outputs C k [b,n,2r]. No full moment matrix is retained.
    """
    z, q, output = finite
    qi, m0qi, m1qi, m2qi, m3qi = infinity
    z = jnp.broadcast_to(z, (q.shape[0], q.shape[-1]))[:, None, :]
    fi = matmul(qi, output, transa="C")
    q0 = 2 * matmul(m0qi, q, transa="C")
    q1 = 2 * matmul(m1qi, q, transa="C")
    g = jnp.concatenate((fi, fi * z - q0), axis=-2)
    h = jnp.concatenate((fi * z - q0, fi * z * z - q0 * z - q1), axis=-2)
    p0, p1, p2, p3 = (2 * matmul(qi, m, transa="C") for m in (m0qi, m1qi, m2qi, m3qi))
    gii = jnp.concatenate((jnp.concatenate((p0, p1), axis=-1),
                           jnp.concatenate((p1, p2), axis=-1)), axis=-2)
    hii = jnp.concatenate((jnp.concatenate((p1, p2), axis=-1),
                           jnp.concatenate((p2, p3), axis=-1)), axis=-2)
    return g, h, gii, hii, 2 * jnp.concatenate((m0qi, m1qi), axis=-1)


def assemble_ordered_shared_pole_pencil(states, infinity, *, matmul):
    """Assemble G=X^H s3 X, H=X^H M X and O=C X for time-reversal-broken data.

    ``states`` holds ``(z,Q,WQ,dW/dz Q)`` with complex z in Ry, not s=z**2,
    in the paired layout of ``_direction_states(ordered=True)``: every state
    X(z) followed, after all originals, by its mirror X(-z) on the same
    directions. The resolvent-identity column of ``finite_pencil_column`` is
    exact for the linear pencil (z s3 - M) as written. ``infinity`` is None for
    a finite-state bank, else the five panels of
    ``ordered_infinity_pencil_column``. Returns Hermitian G, H, O and the
    finite nodes z [b,R_finite] that the paired reduction needs.
    """
    q = jnp.concatenate([state[1] for state in states], axis=-1)
    output = jnp.concatenate([state[2] for state in states], axis=-1)
    derivative = jnp.concatenate([state[3] for state in states], axis=-1)
    z = jnp.concatenate([jnp.broadcast_to(jnp.asarray(state[0], jnp.complex128),
                         (state[1].shape[0], state[1].shape[-1]))
                         for state in states], axis=-1)
    finite = (z, q, output)
    g, h = finite_pencil_column(finite, (z, q, output, derivative), matmul=matmul)
    if infinity is not None:
        gi, hi, gii, hii, oi = ordered_infinity_pencil_column(finite, infinity, matmul=matmul)
        g = jnp.concatenate((jnp.concatenate((g, _adjoint(gi)), axis=-1),
                             jnp.concatenate((gi, gii), axis=-1)), axis=-2)
        h = jnp.concatenate((jnp.concatenate((h, _adjoint(hi)), axis=-1),
                             jnp.concatenate((hi, hii), axis=-1)), axis=-2)
        output = jnp.concatenate((output, oi), axis=-1)
    return _hermitian(g), _hermitian(h), output, z


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
    import jax

    identity = _diagonal_face(jnp.ones(metric.shape[:1] + metric.shape[-1:]), metric)
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



def reduce_shared_pole_pencil(pencil, active_columns, *, eigh, matmul, gates):
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
    correction, metric_ok, metric_diagnostics = _metric_inverse_root(
        _hermitian(metric) + null_identity, matmul=matmul,
        tolerance=gates["retained_subspace_moments"]["threshold"])
    z = matmul(z, correction) * keep[:, None, :]
    metric = matmul(z, matmul(g, z), transa="C")
    t = _hermitian(matmul(z, matmul(h, z), transa="C"))
    # The norm puts the inert spectrum strictly below every physical Ritz
    # value, including negative physical values which must reach zero policy.
    sentinel = -(jnp.linalg.norm(t, axis=(-2, -1)) + 1)
    t = t + null_identity * sentinel[:, None, None]
    poles, rotation = eigh(t)
    active = jnp.arange(g.shape[-1])[None, :] >= g.shape[-1] - count[:, None]
    b = matmul(matmul(output, z), rotation) * active[:, None, :]
    poles = jnp.where(active, poles, 1.0)
    wanted_metric = _diagonal_face(keep, g)
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
    return _hermitian(jnp.concatenate(
        (jnp.concatenate((ww, wv), axis=-1), jnp.concatenate((_adjoint(wv), vv), axis=-1)), axis=-2))


def _join_columns(a, b):
    return jnp.concatenate((a, b), axis=-1)


@lru_cache(maxsize=None)
def _paired_kernels(face, output_face, half, finite, n_inf):
    """Glue of the paired reduction with every large result on the x/y face.

    Eager slicing, concatenation and a + a^H of face-sharded operands return
    replicated arrays (measured on a 2x2 host mesh,
    runs/frequency_integration_sandbox/425_trint_20260915/logs/eager_sharding_probe.log).
    The same elementwise arithmetic compiled with face output shardings keeps
    each [b,R,R] block at 16 R^2/(Px Py) bytes per rank. Unsharded callers
    (host tests) get the plain functions.
    """
    statics = dict(half=half, finite=finite, n_inf=n_inf)
    member, outputs = partial(_paired_member, **statics), partial(_paired_output, **statics)
    if face is None:
        return member, outputs, _hermitian, _restricted_block, _join_columns
    output_face = face if output_face is None else output_face
    return (jax.jit(member, out_shardings=(face, face, face)),
            jax.jit(outputs, out_shardings=(output_face, output_face)),
            jax.jit(_hermitian, out_shardings=face),
            jax.jit(_restricted_block, out_shardings=face),
            jax.jit(_join_columns, out_shardings=output_face))


def reduce_ordered_shared_pole_pencil(pencil, active_columns, *, eigh, matmul, gates):
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
    is reported.

    Returns (b [b,n,R], poles2 [b,R], active [b,R]), the signed model
    (c [b,n,R], mu [b,R], retained [b,R]) and device diagnostics.
    """
    from jax.sharding import NamedSharding

    g, h, output, points = pencil
    side, finite = int(g.shape[-1]), int(points.shape[-1])
    half, n_inf = finite // 2, (side - finite) // 2
    if finite % 2 or (side - finite) % 2:
        raise ValueError(f"GATE shared_pole_orientation_pair: got: {finite} finite and {side - finite} infinity columns; want: even counts; why: the ordered cut acts in the paired basis")
    paired = (jnp.all(points[:, half:] == -points[:, :half])
              & jnp.all(active_columns[:, half:finite] == active_columns[:, :half])
              & jnp.all(active_columns[:, finite + n_inf:] == active_columns[:, finite:finite + n_inf]))
    if not bool(paired):
        raise ValueError("GATE shared_pole_orientation_pair: got: finite columns not in mirrored halves; want: [X(z); X(-z)] on one direction set and paired k0/k1 columns; why: the ordered cut acts in the paired basis")
    face = g.sharding if isinstance(getattr(g, "sharding", None), NamedSharding) else None
    output_face = output.sharding if isinstance(getattr(output, "sharding", None), NamedSharding) else None
    member, outputs, hermitian, block, join = _paired_kernels(face, output_face, half, finite, n_inf)

    live_f = active_columns[:, :half]
    inverse = jnp.where(live_f, 1 / jnp.where(live_f, 2 * points[:, :half], 1), 0)
    # Every large result below stays on the x/y face; eager slicing, concatenation
    # and a + a^H come out replicated ([b,R,R] per rank), which is what ran CrI3 out of memory.
    g_ww, g_wv, g_vv = member(g, inverse)
    h_ww, h_wv, h_vv = member(h, inverse)
    o_w, o_v = outputs(output, inverse)
    active = jnp.concatenate((live_f, active_columns[:, finite:finite + n_inf]), axis=-1)
    diagonal = jnp.real(jnp.diagonal(h_vv, axis1=-2, axis2=-1))
    diagonal_ok = jnp.all(jnp.where(active, jnp.isfinite(diagonal) & (diagonal > 0), diagonal == 0), axis=-1)
    scale = jnp.where(active, 1 / jnp.sqrt(jnp.where(diagonal > 0, diagonal, 1)), 0)
    sandwich = lambda a: scale[:, :, None] * a * scale[:, None, :]
    g_ww, g_wv, g_vv, h_ww, h_wv, h_vv = (sandwich(a) for a in (g_ww, g_wv, g_vv, h_ww, h_wv, h_vv))
    o_w, o_v = o_w * scale[:, None, :], o_v * scale[:, None, :]
    validity = gates["normalized_gram_validity"]["threshold"]
    gamma, u = eigh(hermitian(h_vv))
    largest = gamma[:, -1]
    ratio = gamma[:, 0] / jnp.where(largest > 0, largest, 1)
    keep = (gamma > gates["normalized_gram_keep"]["threshold"] * largest[:, None]) & (largest[:, None] > 0)
    count = jnp.sum(keep, axis=-1, dtype=jnp.int64)
    z = u * (keep / jnp.sqrt(jnp.where(keep, gamma, 1)))[:, None, :]
    del u
    metric = matmul(z, matmul(h_vv, z), transa="C")
    null_identity = _diagonal_face(~keep, h_vv)
    correction, metric_ok, metric_diagnostics = _metric_inverse_root(
        hermitian(metric) + null_identity, matmul=matmul,
        tolerance=gates["retained_subspace_moments"]["threshold"])
    del null_identity
    z = matmul(z, correction) * keep[:, None, :]
    del correction
    metric = matmul(z, matmul(h_vv, z), transa="C")
    wanted_metric = _diagonal_face(keep, h_vv)
    metric_relative = (jnp.linalg.norm(metric - wanted_metric, axis=(-2, -1))
                       / jnp.sqrt(jnp.maximum(count, 1)))
    del wanted_metric, h_vv
    project = lambda a: matmul(z, matmul(a, z), transa="C")
    # Restricted paired pencil on span(Z) in both halves. Its v-block is the metric
    # (identity after correction). On time-reversal-symmetric data H_r = diag(t_s, I),
    # t_s the even route's Z^H H_s Z; once time reversal is broken the halves mix and a
    # w combination can lie in span(v), so a second relative keep cut on H_r removes
    # exactly those redundant combinations before the H-metric Ritz step.
    h_r = block(project(h_ww), project(h_wv), metric)
    del h_ww, h_wv, metric
    g_r = block(project(g_ww), project(g_wv), project(g_vv))
    del g_ww, g_wv, g_vv
    o_r = join(matmul(o_w, z), matmul(o_v, z))
    del o_w, o_v, z
    gamma_r, u_r = eigh(h_r)
    top_r = gamma_r[:, -1]
    ratio_r = gamma_r[:, 0] / jnp.where(top_r > 0, top_r, 1)
    keep_r = (gamma_r > gates["normalized_gram_keep"]["threshold"] * top_r[:, None]) & (top_r[:, None] > 0)
    count_r = jnp.sum(keep_r, axis=-1, dtype=jnp.int64)
    y = u_r * (keep_r / jnp.sqrt(jnp.where(keep_r, gamma_r, 1)))[:, None, :]
    del u_r
    null_r = _diagonal_face(~keep_r, h_r)
    metric_r = hermitian(matmul(y, matmul(h_r, y), transa="C")) + null_r
    # The restricted sources are released before the second metric correction.
    del null_r, h_r
    correction_r, metric_r_ok, _ = _metric_inverse_root(
        metric_r, matmul=matmul, tolerance=gates["retained_subspace_moments"]["threshold"])
    del metric_r
    y = matmul(y, correction_r) * keep_r[:, None, :]
    del correction_r
    mu, rotation = eigh(hermitian(matmul(y, matmul(g_r, y), transa="C")))
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
    }
    return (b, poles2, positive), (c, mu, retained), diagnostics


def ordered_moment_identity(signed, infinity, *, matmul):
    """Projected z-moments of the signed model against the bank's M0..M3.

    m_n(model) = sum_j c_j c_j^H mu_j^-(n+1) on the ORIGINAL Q_inf, relative
    Frobenius defect of each order against its own |Q^H 2M_n Q|. With k0, k1
    in the full Galerkin span it matches n = 0..3 exactly; after the keep and
    retention cuts only to projection accuracy (diagnostic, like the TRS
    full_m1/full_m3 rows). ``infinity`` is the five-panel tuple; returns {m0..m3: [b]}.
    """
    c, mu, retained = signed
    qi, *moments = infinity
    a = matmul(qi, c, transa="C")
    inverse = jnp.where(retained, 1 / jnp.where(retained, mu, 1), 0)
    rows = {}
    for n, m in enumerate(moments):
        exact = 2 * matmul(qi, m, transa="C")
        value = matmul(a * (inverse ** (n + 1))[:, None, :], a, transb="C")
        scale = jnp.linalg.norm(exact, axis=(-2, -1))
        rows[f"m{n}"] = jnp.linalg.norm(value - exact, axis=(-2, -1)) / jnp.where(scale > 0, scale, 1)
    return rows


def ordered_pole_bound_ry(m1, inverse_coulomb_sqrt, *, energy_span_ry, gap_ry, matmul, eigh):
    """Upper bound on the RPA poles of W from the bank's own quantities, per parent.

    The poles are +-eigenvalues of s3 M, M = M0 + K with M0 = diag(D) the bare
    transitions and K = Phi^H V Phi >= 0, so |Omega| <= ||M|| <= D_max + ||K||.
    Every term of A0 = sum_j D_j (P_j + conj P_j) is PSD with D_j >= D_min, so
    ||K|| <= ||x2|| / D_min with x2 = H A0 H, and x2 <= H^+ 2M1 H^+ (the
    difference is x1^2 >= 0). Hence Omega_max <= D_max + lambda_max(H^+ 2M1 H^+)/D_min.
    ``energy_span_ry`` >= D_max and ``gap_ry`` <= D_min come from the bank census;
    a gapless census returns +inf and the bound never fires. ``m1`` and
    ``inverse_coulomb_sqrt`` are [b,n,n] face arrays; returns [b] float64 (Ry).
    """
    if not float(gap_ry) > 0:
        return jnp.full((m1.shape[0],), jnp.inf)
    whitened = _hermitian(matmul(inverse_coulomb_sqrt, matmul(2 * m1, inverse_coulomb_sqrt)))
    top = eigh(whitened)[0][:, -1]
    return float(energy_span_ry) + jnp.maximum(top, 0) / float(gap_ry)


def ordered_shared_pole_value(model, partner, z, *, matmul):
    """Evaluate the ordered carrier Wc_p(z) from stored positive-pole factors.

    ``model`` and ``partner`` are (b [b,n,K], poles2 [b,K], active [b,K]) for
    parent p and for the parent of -q (the same tuple at q = -q). Returns
    b diag(1/(2W(z-W))) b^H - conj(b~) diag(1/(2W~(z+W~))) b~^T. At p = -q
    this is sum Re(bb^H)/(z^2-W^2) + i Im(bb^H) z/(W(z^2-W^2)): the odd channel
    is one extra contraction of the same vector, with no extra storage.
    """
    b, poles2, active = model
    bt, poles2t, activet = partner
    w = jnp.sqrt(jnp.where(active, poles2, 1.0))
    wt = jnp.sqrt(jnp.where(activet, poles2t, 1.0))
    d = jnp.where(active, 1 / (2 * w * (z - w)), 0)
    dt = jnp.where(activet, 1 / (2 * wt * (z + wt)), 0)
    return (matmul(b * d[:, None, :], b, transb="C")
            - matmul(bt.conj() * dt[:, None, :], bt.conj(), transb="C"))


def signed_shared_pole_passivity(signed, inverse_coulomb_sqrt, *, eta_ry, matmul, eigh, gates):
    """Hermitian-part passivity of the signed particle-hole model at z = i eta.

    -Wc_r(i eta) = sum_j c_j c_j^H/(1 - i eta mu_j); its Hermitian part sums
    both signs of real frequency and lies in [0, I] after V whitening for a
    stable response. The anti-Hermitian part is the odd channel: reported.
    """
    c, mu, retained = signed
    whitened = matmul(inverse_coulomb_sqrt, c)
    weight = jnp.where(retained, 1 / (1 - 1j * eta_ry * mu), 0)
    response = matmul(whitened * weight[:, None, :], whitened, transb="C")
    return _passivity_response_checks(response, eigh=eigh, gates=gates)


def retained_moment_identity(pencil, coefficients, model, infinity_selector, *, matmul):
    """Check the two moment blocks on projected latent infinity states.

    Let E select infinity columns of X, G=X.H X, H=X.H T X, and Y be
    the active Ritz coefficient map after both Gram and zero-policy cuts.
    A=Y.H G E and B=Y A represent P_ret X_inf = X B. The independent
    original-pencil targets are B.H G B / 2 and B.H H B / 2; the model
    values are A.H A / 2 and A.H Lambda A / 2. These compare the same
    retained latent states, not the discarded original infinity components.

    ``pencil=(G,H,O)`` has face tiles [b,R,R]/[b,n,R]; coefficients is
    [b,R,Kp], model=(b,Lambda,active), and infinity_selector E [b,R,r_inf].
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

    ``model=(b,poles2,active)`` has shapes [b,n,Kp], [b,Kp], [b,Kp].
    The weight is sum_j ||b_j||**2 (Ry**3), excluding carrier sentinels.
    A failed predicate must refuse before export; no pole is clipped.
    """
    b, poles, active = model
    drop = active & (poles <= gates["zero_ritz_policy"]["threshold"]["lambda_cutoff_ry2"])
    keep = active & ~drop
    weights = jnp.sum(jnp.abs(b) ** 2, axis=-2)
    total = jnp.sum(jnp.where(active, weights, 0), axis=-1)
    lost = jnp.sum(jnp.where(drop, weights, 0), axis=-1)
    fraction = lost / jnp.where(total > 0, total, 1)
    count = jnp.sum(keep, axis=-1, dtype=jnp.int64)
    finite = jnp.all(jnp.isfinite(b), axis=(-2, -1)) & jnp.all(jnp.isfinite(poles), axis=-1)
    admitted = (finite & (total > 0) & (count > 0)
                & (fraction <= gates["zero_ritz_policy"]["threshold"]["max_dropped_weight_fraction"]))
    return (jnp.where(keep[:, None, :], b, 0), jnp.where(keep, poles, 1), keep), {
        "zero_policy": admitted,
        "dropped_factor_weight_fraction": fraction,
        "dropped_count": jnp.sum(drop, axis=-1, dtype=jnp.int64),
        "factor_weight": total,
        "retained_rank": count,
    }


@lru_cache(maxsize=None)
def _factor_column_permutation(mesh):
    """Stream a joint pole-column permutation through one y tile at a time.

    A global gather of b replicates the complete K axis on each x row.
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
    """Sort joint active b/Lambda columns, retaining ties and safe padding.

    Returns the same three model arrays and the replicated [b,Kp]
    permutation. Active entries form a prefix, inactive b is exactly zero,
    inactive Lambda is 1 Ry**2. Stable sorting preserves equal-pole order.
    """
    b, poles, active = model
    order = jnp.argsort(jnp.where(active, poles, jnp.inf), axis=-1, stable=True)
    b = _factor_column_permutation(mesh_xy)(b, order)
    poles = jnp.take_along_axis(poles, order, axis=-1)
    active = jnp.take_along_axis(active, order, axis=-1)
    return (jnp.where(active[:, None, :], b, 0), jnp.where(active, poles, 1), active), order


def shared_pole_passivity(model, inverse_coulomb_sqrt, *, eta_ry, matmul, eigh, gates):
    """Test raw latent-model passivity on the authenticated Coulomb support.

    The authenticated inverse square root [b,n,n] and b [b,n,Kp] are
    face-tiled complex128. Lambda/active [b,Kp] are replicated. Returns
    device scalars; all ranks must evaluate them before the host refusal.
    """
    b, poles, active = model
    whitened = matmul(inverse_coulomb_sqrt, b)
    weight = jnp.where(active, 1 / (poles + eta_ry**2), 0)
    response = matmul(whitened * weight[:, None, :], whitened, transb="C")
    return _passivity_response_checks(response, eigh=eigh, gates=gates)


def shared_pole_operator_passivity(wc, inverse_coulomb_sqrt, *, matmul, eigh, gates):
    """Test an evaluated physical Wc(i eta) against the actual V support.

    ``wc`` may include the versioned symmetry realization; no factors of its
    averaged residues need to be materialized. Both square operands retain
    their caller's all-P faces. The caller owns evaluation at imaginary eta
    and memory admission, including the two square GEMMs and eigensolve.
    """
    inverse = inverse_coulomb_sqrt
    response = -matmul(matmul(inverse, wc), inverse)
    return _passivity_response_checks(response, eigh=eigh, gates=gates)


def _passivity_response_checks(response, *, eigh, gates):
    """Common 0 <= whitened response <= I and Hermiticity gate."""
    herm = _hermitian(response)
    norm = jnp.linalg.norm(herm, axis=(-2, -1))
    anti = jnp.linalg.norm(response - _adjoint(response), axis=(-2, -1))
    anti = anti / jnp.where(norm > 0, 2 * norm, 1)
    values, _ = eigh(herm)
    minimum, maximum = values[:, 0], values[:, -1]
    limit = gates["passivity"]["threshold"].get("antihermitian_relative_max")
    passed = ((minimum >= gates["passivity"]["threshold"]["eigenvalue_min"])
              & (maximum <= gates["passivity"]["threshold"]["eigenvalue_max"])
              & (True if limit is None else anti <= limit)
              & jnp.all(jnp.isfinite(values), axis=-1) & jnp.isfinite(anti))
    return {"passivity": passed, "passivity_min": minimum,
            "passivity_max": maximum, "passivity_antihermitian_relative": anti}


def shared_pole_reciprocity(value, reference, *, gates):
    """Check W(s).T=W(s) where held input data have this extra symmetry.

    ``value`` and ``reference`` are matching [...,n,n] complex128 response
    faces (W in Ry or dW/ds in inverse Ry). Leading axes label parents and
    samples. Returns scalar arrays on those axes; no matrix is retained.
    For a real-residue Stieltjes response, transpose symmetry off the real
    s axis is equivalent to entrywise realness on the negative s axis.
    Scalar TRS alone does not impose this symmetry at a generic fixed q.
    This is a sampled model gate, not a certificate at every frequency.
    """
    threshold = gates["model_reciprocity"]["threshold"]
    def defect(a):
        return jnp.linalg.norm(a-jnp.swapaxes(a, -1, -2), axis=(-2, -1)) / jnp.maximum(
            jnp.linalg.norm(a, axis=(-2, -1)), jnp.finfo(jnp.float64).tiny)
    exact, measured = defect(reference), defect(value)
    applicable = exact <= threshold["reference_relative_max"]
    passed = (jnp.isfinite(exact) & jnp.isfinite(measured)
              & (~applicable | (measured <= threshold["model_relative_max"])))
    return {"passed": passed, "applicable": applicable,
            "reference_relative": exact, "model_relative": measured}


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


@lru_cache(maxsize=256)
def _parent_panel_slice(mesh_xy, width):
    """Slice [b,n,r] direction/action faces without a host or replicated seam."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    return jax.jit(
        lambda arrays, parent: jax.tree.map(
            lambda a: jax.lax.dynamic_slice_in_dim(a, parent, 1, axis=0)[:, :, :width], arrays),
        out_shardings=NamedSharding(mesh_xy, P(None, 'x', 'y')))



@lru_cache(maxsize=None)
def _parent_result_slice(mesh_xy):
    """Slice a parent's padded factor and scalar receipts in one executable."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    face = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    scalar = NamedSharding(mesh_xy, P())
    return jax.jit(
        lambda result, parent: jax.tree.map(
            lambda a: jax.lax.dynamic_slice_in_dim(a, parent, 1, axis=0), result),
        out_shardings=((face, scalar, scalar), scalar, scalar, scalar))


@lru_cache(maxsize=None)
def _hermitian_part_kernel(mesh):
    """Reuse Hermitian projection on current [b,n,n] response faces."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    return jax.jit(_hermitian, out_shardings=NamedSharding(mesh, P(None, 'x', 'y')))


@lru_cache(maxsize=None)
def _public_factor_kernel(mesh):
    """Insert the scalar-spin axis without recreating the executable."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    return jax.jit(lambda value: value[:, :, None, :],
                   out_shardings=NamedSharding(mesh, P(None, 'x', None, 'y')))


@lru_cache(maxsize=None)
def _stack_model_kernel(mesh):
    """Stack the current admitted factor/pole/count batch on named layouts."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    return jax.jit(
        lambda parts: tuple(jnp.concatenate([row[i] for row in parts], axis=0)
                            for i in range(3)),
        out_shardings=(NamedSharding(mesh, P(None, 'x', None, 'y')),
                       NamedSharding(mesh, P()), NamedSharding(mesh, P())))


def _odd_partner_directions(q, output, cutoff, *, eigh_plan, matmul, column_extent, multiplet_tol):
    """Partner directions of O = W Q orthogonal to Q above the recipe direction cutoff.

    For a role whose conjugate node brings no new tangent under time reversal
    (imaginary z, or a line role with Re z = 0), W Q lies in span(Q) on
    time-reversal-symmetric data and the partner would only duplicate Q. The
    component of O orthogonal to Q is rank-revealed against the largest
    singular value of O with the same relative cutoff used for line directions;
    on a magnet the surviving directions carry the odd channel. Only spectra
    cross to the host. Returns (orthonormal directions [b,n,r], widths) or
    (None, None) when nothing survives.
    """
    import distrib_la
    import numpy as np

    hermitian_part = _hermitian_part_kernel(eigh_plan.mesh)
    remainder = output - matmul(q, matmul(q, output, transa="C"))
    top = np.asarray(eigh_plan.batched(hermitian_part(matmul(output, output, transb="C")))[0])[..., -1]
    perp = hermitian_part(matmul(remainder, remainder, transb="C"))
    values = np.asarray(eigh_plan.batched(perp)[0])[..., ::-1]
    counts = [int(np.sum(row > float(cutoff) ** 2 * float(t)))
              for row, t in zip(np.atleast_2d(values), np.atleast_1d(top))]
    if max(counts) == 0:
        return None, None
    directions, kept = distrib_la.leading_eigenvectors(
        perp, max(counts), eigh_plan=eigh_plan, column_extent=column_extent, multiplet_tol=multiplet_tol)
    return directions, tuple(int(v.shape[-1]) for v in kept)


def _direction_states(read_sample, recipe, *, eigh_plan, svd_plan, matmul,
                      column_extent, logical_n, admit, infinity_carrier, ordered=False,
                      read_mirror=None):
    """Select each q row independently from a bounded batch of fitted samples.

    ``read_sample`` returns W/dW [b,n,n] faces. Only spectra cross the host;
    Output states keep their batch axis. Counts and role receipts are small
    host metadata; the packer compacts each parent's original port carriers.

    Ordered data pair every state X(z) on direction Q with its particle-hole
    mirror X(-z) on the SAME Q, the layout ``reduce_ordered_shared_pole_pencil``
    cuts in. The mirror needs W(-conj z) and dW/ds(-conj z): for purely
    imaginary z it is the sample itself; otherwise ``read_mirror(id)``
    (production: the conjugated sample of the parent of -q) or, when None, the
    one fitted recipe id at -conj(z), which then selects no directions of its
    own. Mirrors follow all original states in the same order.
    """
    import distrib_la
    import numpy as np

    hermitian_part = _hermitian_part_kernel(eigh_plan.mesh)
    fit_roles = _fit_roles(recipe)
    fit_ids = [int(i) for i in recipe["fit_ids"]]
    mirror_source = {}
    if ordered and read_mirror is None:
        points = {i: _sample_point(recipe, i) for i in fit_ids}
        for j in fit_ids:
            # The first fitted id of a pair is the original; its -conj(z) partner is the mirror.
            if points[j].real == 0 or j in mirror_source.values():
                continue
            matches = [k for k in fit_ids if points[k] == -points[j].conjugate()]
            if len(matches) != 1:
                raise ValueError(f"GATE shared_pole_orientation_pair: got: {len(matches)} fitted supports at -conj(z) for sample {j}; want: exactly one mirror sample or read_mirror; why: the ordered cut acts on paired particle-hole states")
            mirror_source[j] = matches[0]
    skip = set(mirror_source.values())
    states, counts, roles, conjugates = [], [], [], []
    mirrors, mirror_counts, mirror_roles = [], [], []
    def largest_side():
        return infinity_carrier + sum(st[1].shape[-1] for st in (*states, *mirrors))
    for sample_id in fit_ids:
        if sample_id in skip:
            continue
        admit(largest_side())
        w, derivative = read_sample(int(sample_id), states)
        if not roles:
            roles = [[] for _ in range(w.shape[0])]
            mirror_roles = [[] for _ in range(w.shape[0])]
        first = len(states)
        for role in fit_roles:
            if role["held"] or int(role["sample_id"]) != int(sample_id):
                continue
            kind = role["role"].split(":", 1)[0]
            if kind == "line":
                q_batch, values = distrib_la.right_singular_vectors(
                    w, recipe["direction_cutoff"], eigh_plan=svd_plan,
                    column_extent=column_extent,
                    multiplet_tol=recipe["multiplet_relative_tolerance"])
            elif kind == "imaginary":
                width = min(logical_n, max(1, int(recipe["imaginary_width"])))
                q_batch, values = distrib_la.leading_eigenvectors(
                    hermitian_part(-w), width, eigh_plan=eigh_plan,
                    column_extent=column_extent,
                    multiplet_tol=recipe["multiplet_relative_tolerance"])
            else:
                raise ValueError(f"GATE shared_pole_role: got: {kind}; want: line or imaginary fitted role; why: unknown tangent semantics")
            widths = tuple(int(v.shape[-1]) for v in values)
            if any(width < 1 or width > logical_n for width in widths):
                raise ValueError(f"GATE shared_pole_directions: got: ranks {widths}; want: 1..{logical_n}; why: empty or padded physical direction set")
            s = (_sample_point(recipe, int(sample_id)) if ordered
                 else _sample_point(recipe, int(sample_id)) ** 2)
            # W(s*)=W(s).H: its right singular space is the LEFT space
            # of W(s). Use WQ=U sigma without another selection or transport.
            # Column equilibration removes sigma; this also preserves the
            # conjugate space when the underlying response is real symmetric.
            # Ordered data: nodes are z; every role takes its conjugate partner
            # (Wc(conj z) = Wc(z)^H holds without time reversal) and actions
            # use dW/dz = 2 z dW/ds.
            for conjugate in ((False, True) if (ordered or kind == "line") and s.imag != 0 else (False,)):
                admit(largest_side() + q_batch.shape[-1])
                transa = "C" if conjugate else "N"
                state_widths = widths
                if conjugate and ordered and (kind == "imaginary" or s.real == 0):
                    # Dedupe before the cut: only partner directions orthogonal to Q survive.
                    direction, state_widths = _odd_partner_directions(
                        states[-1][1], states[-1][2], recipe["direction_cutoff"], eigh_plan=eigh_plan,
                        matmul=matmul, column_extent=column_extent,
                        multiplet_tol=recipe["multiplet_relative_tolerance"])
                    if direction is None:
                        continue
                else:
                    direction = states[-1][2] if conjugate else q_batch
                output = matmul(w, direction, transa=transa)
                action = matmul(derivative, direction, transa=transa)
                if ordered:
                    action = action * (2 * (s.conjugate() if conjugate else s))
                states.append((s.conjugate() if conjugate else s, direction, output, action))
                conjugates.append(conjugate)
                counts.append(state_widths)
                for i, width in enumerate(state_widths):
                    roles[i].append({"sample_id": int(sample_id), "role": role["role"],
                                     "conjugate": conjugate, "width": width,
                                     "carrier_width": column_extent(width)})
            del q_batch, direction, output, action
        if ordered and len(states) > first:
            if _sample_point(recipe, int(sample_id)).real == 0:
                w_m, d_m = w, derivative
            elif read_mirror is not None:
                w_m, d_m = read_mirror(int(sample_id))
            else:
                w_m, d_m = read_sample(mirror_source[int(sample_id)], states)
            # W(-z) = W(-conj z)^H on Q; W(-conj z) on the partner's O; in both
            # cases dW/dz at the mirror node is -2 node dW/ds(-conj z) (adjoint on Q).
            for index in range(first, len(states)):
                node, direction = states[index][0], states[index][1]
                admit(largest_side() + direction.shape[-1])
                transa = "N" if conjugates[index] else "C"
                mirrors.append((-node, direction, matmul(w_m, direction, transa=transa),
                                matmul(d_m, direction, transa=transa) * (-2 * node)))
                mirror_counts.append(counts[index])
                for i in range(len(roles)):
                    mirror_roles[i].append(dict(roles[i][index], mirror=True))
            del w_m, d_m
        del w, derivative
    if ordered:
        states, counts = states + mirrors, counts + mirror_counts
        roles = [row + mirror_row for row, mirror_row in zip(roles, mirror_roles)]
    return states, np.asarray(counts, dtype=np.int64).T, roles


def _model_diagnostics(model, moments, infinity_directions, *, matmul):
    """Compare M1=bb.H/2 and M3=b Lambda b.H/2 with physical moments.

    Factors are [b,n,K], moments [b,n,n], and infinity directions [b,n,r],
    in their existing face layouts. The loop selects from the two resident
    moments without stacking dense matrices; only [2,b] scalar defects are
    stacked. The caller compiles this stage with its accounted service GEMM.
    """
    b, poles, _ = model

    def moment_defect(third):
        target = jax.lax.cond(third, lambda: moments["M3"],
                              lambda: moments["M1"])
        weighted = jax.lax.cond(third, lambda: b * poles[:, None, :],
                                lambda: b)
        value = matmul(weighted, b, transb="C") / 2
        defect = target - value
        projected = matmul(infinity_directions,
                           matmul(defect, infinity_directions), transa="C")
        projected_target = matmul(infinity_directions,
                                  matmul(target, infinity_directions), transa="C")
        norm = jnp.linalg.norm(target, axis=(-2, -1))
        projected_norm = jnp.linalg.norm(projected_target, axis=(-2, -1))
        return (jnp.linalg.norm(defect, axis=(-2, -1)) / norm,
                jnp.linalg.norm(projected, axis=(-2, -1)) / projected_norm)

    full, projected = jax.lax.map(moment_defect, jnp.asarray([False, True]))
    return {
        "M1": {"full_relative": full[0], "original_infinity_relative": projected[0]},
        "M3": {"full_relative": full[1], "original_infinity_relative": projected[1]},
    }


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
    timing.fence("spole.entry")
    with timing.section("spole.entry"):
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
            shared_real_pole_gates_ordered_v1,
        )
        from common.units import RYD_TO_EV
        from gw.w_isdf import response_coulomb_powers

    timing.fence("spole.setup")
    with timing.section("spole.setup"):
        recipe = meta.shared_pole_recipe
        # Time-reversal-broken scalar states take the ordered particle-hole route.
        ordered = not bool(bank["tables"]["sym"].trs_allowed)
        from file_io.shared_pole_store import charge_representation
        # Scalar and two-component decks share one mu x mu charge operator.
        if not charge_representation(meta):
            raise ValueError("GATE shared_pole_representation: got: bispinor or unsupported state; want: scalar or two-component charge operator (TRS-broken states take the ordered route); why: both-endpoint spin action is not yet supported")
        if ordered:
            gates = shared_real_pole_gates_ordered_v1
        resolution = linalg_resolution({"linalg": config.backend.linalg})
        identity = bank["identity"]
        ledger = meta.shared_pole_capacity
        upstream = ledger.live_stages
        n = int(meta.n_rmu_padded)
        native_queries = {}
        native_maxima = {"eigh": 0, "gemm": 0}
        workspace = 0
        current_side = 0
        current_phase = "selection"
        pending = []
        retained_panels = ()
        batch_width = 1
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
                size = (distrib_la.matmul_workspace_bytes_per_rank(
                    mesh_xy, shapes, np.complex128, backend="auto",
                    batched_route=resolution.batched_route) if op == "gemm" else
                    distrib_la.workspace_bytes_per_rank(plan, op, shapes, np.complex128))
                native_queries[key] = size
            if op == "gemm":
                native_maxima[op] = max(native_maxima[op], native_queries[key])
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
        if bool(header.get("ordered", False)) != ordered:
            raise ValueError(f"GATE shared_pole_representation: got: bank ordered={bool(header.get('ordered', False))} with trs_allowed={not ordered}; want: an ordered bank exactly when time reversal is broken (both orientations, -q as its own parent); why: particle-hole pairing across parents")
        # Odd z-moments M0/M2 certify the ordered infinity block; a finite-state
        # ordered bank builds without it and records the moments NOT_MEASURED.
        odd_moments = ordered and bool(header.get("odd_moments", False))

        def capacity(side, *, phase=None, sample_batch=1, transpose_staging=0):
            nonlocal current_side, current_phase, workspace
            if phase is not None:
                current_phase = phase
            current_side = side
            extents = {n, 2*n} if current_phase == "selection" else (
                {side} if current_phase == "reduction" else {n})
            # Eigh scratch is transient: replace it at each phase boundary.
            # Only the actually used GEMM context workspace persists.
            native_maxima["eigh"] = max(query_workspace(
                "eigh", ((batch_width, extent, extent),), eigenplan(extent))
                for extent in sorted(extents))
            workspace = sum(native_maxima.values())
            price = shared_pole_byte_terms(
                meta, mesh_xy=mesh_xy, resolution=resolution, pencil_side=side,
                parent_batch=batch_width, sample_batch=sample_batch, phase=current_phase)
            # Other parents' narrow inputs survive selection and each model's
            # checks; they are additional live storage, never hidden in a limit.
            extra = sum(int(np.prod(a.sharding.shard_shape(a.shape))) * a.dtype.itemsize
                        for a in {id(a): a for a in retained_panels}.values())
            price["terms_bytes_per_rank"]["retained_parent_panels"] = extra
            price["terms_bytes_per_rank"]["gemm_transpose_staging"] = transpose_staging
            price["resident_bytes_per_rank"] += extra + transpose_staging
            row = ledger.reserve(f"constructor.plan.{len(ledger.entries)}",
                                 resident_bytes_per_rank=price["resident_bytes_per_rank"],
                                 workspace_bytes_per_rank=workspace,
                                 concurrent_with=upstream)
            return dict(row, price=price, native_workspace=dict(native_maxima))

        def expose_live(arrays):
            # Callees price only their additional allocations. Supply their
            # exact current inputs instead of the future dense-phase envelope,
            # so a store read does not count its own returned arrays twice.
            unique = {id(array): array for array in (*arrays, *retained_panels)}
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
            # Workspace belongs to the matmul route, not the eigh backend.
            # Its query excludes operand-sized endpoint transpose staging;
            # charge those transient faces separately before execution.
            shapes = tuple(value.shape[:-2] + (value.shape[-2:][::-1]
                           if kwargs.get(trans, "N") != "N" else value.shape[-2:])
                           for value, trans in ((a, "transa"), (b, "transb")))
            previous = workspace
            query_workspace("gemm", shapes, None)
            staging = (sum(int(np.prod(value.shape)) * value.dtype.itemsize
                           // int(mesh_xy.size)
                           for value, trans in ((a, "transa"), (b, "transb"))
                           if kwargs.get(trans, "N") != "N")
                       if resolution.batched_route != "batch_reshard" else 0)
            if workspace != previous or staging:
                capacity(current_side, transpose_staging=staging)
            return distrib_la.matmul(a, b, mesh=mesh_xy, backend="auto",
                                     batched_route=resolution.batched_route, **kwargs)

        eig, svd = eigenplan(n), eigenplan(2*n)
        # Bind once for this construction, outside both parent and support
        # loops. The GEMM closure owns native workspace accounting; keeping
        # this jit local also avoids retaining the map's capacity ledger in
        # a global callable cache across self-consistent reconstructions.
        model_diagnostics = jax.jit(partial(_model_diagnostics, matmul=mm))
        receipts = []
        receipt_entry_start = 0
        stack_models = _stack_model_kernel(mesh_xy)
        from runtime.padding import mesh_divisor
        from gw.shared_pole_local import pack_parent_panels, local_parent_reducer
        # Local dense algebra assigns independent parents to mesh ranks. The
        # distributed plan keeps its one-parent face-tiled execution schedule.
        # The ordered route runs the one-parent distributed schedule.
        local_layout = resolution.layout == "local" and not ordered
        batch_limit = mesh_divisor(mesh_xy) if local_layout else 1
        nq = int(header["bank_shape"]["nq"])
    for q_start in range(0, nq, batch_limit):
        timing.fence("spole.batch_admission")
        with timing.section("spole.batch_admission"):
            q_stop = min(q_start + batch_limit, nq)
            span = (q_start, q_stop)
            batch_width = q_stop - q_start
            capacity(0, phase="selection")
            expose_live(())
        timing.fence("spole.scratch_read")
        with timing.section("spole.scratch_read"):
            with SlabIO(moments["path"], mode="r", mesh=mesh_xy) as moment_io:
                exact = read_shared_pole_bank(moment_io, span, meta=meta,
                                              header=moment_header,
                                              fields=("M0", "M1", "M2", "M3") if odd_moments else ("M1", "M3"))
        timing.fence("spole.infinity_selection")
        with timing.section("spole.infinity_selection"):
            width = min(logical_n, max(1, int(recipe["infinity_width"])))
            qi, infinity_values = distrib_la.leading_eigenvectors(
                exact["M1"], width, eigh_plan=eig, column_extent=column_extent,
                multiplet_tol=recipe["multiplet_relative_tolerance"])
            capacity(qi.shape[-1], phase="selection")
            infinity = ((qi, mm(exact["M0"], qi), mm(exact["M1"], qi), mm(exact["M2"], qi), mm(exact["M3"], qi))
                        if odd_moments else (qi, mm(exact["M1"], qi), mm(exact["M3"], qi)))
            del exact
        with SlabIO(bank["path"], mode="r", mesh=mesh_xy) as bank_io:
            timing.fence("spole.sample_batch_read")
            with timing.section("spole.sample_batch_read"):
                sample_lo = min(int(i) for i in recipe["fit_ids"])
                sample_hi = max(int(i) for i in recipe["fit_ids"]) + 1
                expose_live(infinity)
                samples = read_shared_pole_bank(
                    bank_io, span, meta=meta, header=header,
                    sample_span=(sample_lo, sample_hi))
                # The store admits this complete bounded scratch batch before
                # allocation. Charge it while directions/actions are selected;
                # release it before admitting the dense pencil.
                retained_panels = tuple(samples.values())
                def read_sample(sample_id, retained_states):
                    index = sample_id - sample_lo
                    return samples["Wc"][:, index], samples["dWc_ds"][:, index]

                read_mirror = None
                if ordered:
                    from gw.mpa.sigma import shared_pole_minus_q_index
                    parents_full = [int(v) for v in header["q_irr_full_idx"]]
                    minus_full = shared_pole_minus_q_index(tuple(int(v) for v in header["grid"]))
                    partner = parents_full.index(int(minus_full[parents_full[q_start]]))

                    def read_mirror(sample_id):
                        # W_q(-conj z) = conj W_-q(z), and dW/ds likewise: one sample of the -q parent.
                        part = read_shared_pole_bank(
                            bank_io, (partner, partner + 1), meta=meta, header=header,
                            sample_span=(int(sample_id), int(sample_id) + 1))
                        return jnp.conj(part["Wc"][:, 0]), jnp.conj(part["dWc_ds"][:, 0])

            timing.fence("spole.direction_selection")
            with timing.section("spole.direction_selection"):
                states, counts, roles = _direction_states(
                    read_sample, recipe, eigh_plan=eig, svd_plan=svd, matmul=mm,
                    column_extent=column_extent, logical_n=logical_n, admit=capacity,
                    infinity_carrier=qi.shape[-1], ordered=ordered, read_mirror=read_mirror)
        timing.fence("spole.direction_pack_and_drain")
        with timing.section("spole.direction_pack_and_drain"):
            del samples
            jax.block_until_ready((states, infinity))
            retained_panels = (*infinity, *(v for st in states for v in st[1:]))
            finite_width = max(sum(row['carrier_width'] for row in parent) for parent in roles)
            infinity_width = infinity[0].shape[-1]
            batch_width = mesh_divisor(mesh_xy) if local_layout else 1
            # The permutation temporarily has the sum of the batched port
            # carriers, before compaction to the largest original parent side.
        timing.fence("spole.reduction_admission")
        with timing.section("spole.reduction_admission"):
            price = capacity(sum(st[1].shape[-1] for st in states) + infinity_width,
                             phase="reduction")
        timing.fence("spole.panel_pack")
        with timing.section("spole.panel_pack"):
            packed, extents = pack_parent_panels(
                states, infinity, counts, [v.shape[-1] for v in infinity_values],
                mesh_xy=mesh_xy, parent_batch=batch_width,
                layout=resolution.layout if not ordered else "distributed")
            for i, values in enumerate(infinity_values):
                ri = column_extent(values.shape[-1])
                parent_infinity = _parent_panel_slice(mesh_xy, ri)(infinity, np.int32(i))
                pending.append((q_start+i, None, None, None, parent_infinity[0], roles[i]))
            del states, infinity, qi, infinity_values, counts, roles, parent_infinity
            retained_panels = (*jax.tree.leaves(packed), *(item[4] for item in pending))
            batch_results = None
        if local_layout:
            timing.fence("spole.reduction_admission")
            with timing.section("spole.reduction_admission"):
                price = capacity(finite_width + infinity_width, phase="reduction")
                reduce_eigh = eigenplan(finite_width + infinity_width)
                extents += (extents[-1],) * (batch_width - len(extents))
            timing.fence("spole.gram_reduction")
            with timing.section("spole.gram_reduction"):
                batch_results = local_parent_reducer(
                    mesh_xy, reduce_eigh.native_fn, extents)(*packed)
                jax.block_until_ready(batch_results)
                del packed
                # Drop selected action panels after the fused boundary. Model
                # slices below may coexist with the full padded result buffer.
                retained_panels = (*jax.tree.leaves(batch_results),
                                   *(item[4] for item in pending))
        else:
            timing.fence("spole.panel_pack")
            with timing.section("spole.panel_pack"):
                # The distributed schedule admits one parent; its packed panels
                # already have the original finite and infinity extents.
                iq, _, _, _, directions, row_roles = pending[0]
                finite, infinity, active, _columns = jax.tree.map(lambda a: a[:1], packed)
                if ordered:
                    # k0 and k1 double the infinity columns; a finite-state bank has none.
                    rf = active.shape[-1] - infinity[0].shape[-1]
                    active = (jnp.concatenate((active, active[:, rf:]), axis=-1)
                              if odd_moments else active[:, :rf])
                    infinity = infinity if odd_moments else None
                pending = [(iq, [finite], infinity, active, directions, row_roles)]
                del packed
        batch_checks = None
        if batch_results is not None:
            timing.fence("spole.reduction_admission")
            with timing.section("spole.reduction_admission"):
                from gw.shared_pole_local import local_model_checks
                check_span = (pending[0][0], pending[-1][0]+1)
                held_ids = tuple(int(i) for i in recipe["held_ids"])
                sample_lo, sample_hi = min(held_ids), max(held_ids)+1
                capacity(batch_results[0][0].shape[-1], phase="model",
                         sample_batch=sample_hi-sample_lo)
                expose_live(batch_results[0])
            timing.fence("spole.coulomb")
            with timing.section("spole.coulomb"):
                coulomb_sqrt, inverse_sqrt, batch_coulomb = response_coulomb_powers(
                    meta, config, mesh_xy=mesh_xy, bank_io=bank, q_span=check_span)
                del coulomb_sqrt
            timing.fence("spole.sample_batch_read")
            with timing.section("spole.sample_batch_read"):
                expose_live((*batch_results[0], inverse_sqrt))
                with SlabIO(bank["path"], mode="r", mesh=mesh_xy) as bank_io:
                    held_samples = read_shared_pole_bank(
                        bank_io, check_span, meta=meta, header=header,
                        sample_span=(sample_lo, sample_hi), fields=("Wc", "dWc_ds"))
            timing.fence("spole.passivity_held")
            with timing.section("spole.passivity_held"):
                indices = jnp.asarray([i-sample_lo for i in held_ids])
                supports = jnp.asarray([_sample_point(recipe, i)**2 for i in held_ids])
                batch_checks = local_model_checks(mesh_xy, eig.native_fn)(
                    batch_results[0], inverse_sqrt,
                    held_samples["Wc"][:, indices], held_samples["dWc_ds"][:, indices],
                    supports, jnp.asarray(recipe["eta_ev"] / RYD_TO_EV))
                batch_checks = jax.tree.map(np.asarray, batch_checks)
                del inverse_sqrt, held_samples, indices, supports
        batch_width = 1
        selected = pending
        pending = []
        ready_models = []
        for slot, (q, states, infinity, active_columns, qi, roles) in enumerate(selected):
            timing.fence("spole.gram_reduction")
            with timing.section("spole.gram_reduction"):
                span = (q, q + 1)
                if batch_results is None:
                    price = capacity(active_columns.shape[-1], phase="reduction")
                    if ordered:
                        pencil = assemble_ordered_shared_pole_pencil(states, infinity, matmul=mm)
                        reduce_eigh = eigenplan(pencil[0].shape[-1])
                        model, signed, reduction = reduce_ordered_shared_pole_pencil(
                            pencil, active_columns, eigh=reduce_eigh.batched, matmul=mm, gates=gates)
                        ordered_retained = (ordered_moment_identity(signed, infinity, matmul=mm)
                                            if odd_moments else {})
                    else:
                        pencil = assemble_shared_pole_pencil(states, infinity, matmul=mm)
                        reduce_eigh = eigenplan(pencil[0].shape[-1])
                        model, reduction, coefficients = reduce_shared_pole_pencil(
                            pencil, active_columns, eigh=reduce_eigh.batched, matmul=mm, gates=gates)
                else:
                    model, reduction, zero, retained = _parent_result_slice(mesh_xy)(
                        batch_results, np.int32(slot))
                del states, infinity
            timing.fence("spole.gates")
            with timing.section("spole.gates"):
                for name in ("gram_diagonal_positive", "gram_valid", "retained_metric_positive"):
                    if not bool(jnp.all(reduction[name])):
                        raise ValueError(
                            f"GATE shared_pole_{name}: got: failed at q={q}, "
                            f"Gram min/max={reduction['gram_min_relative']}, "
                            f"metric infinity norm={reduction['metric_initial_infinity_norm']}, "
                            f"inverse-root residual={reduction['metric_inverse_root_residual_relative']}; "
                            f"want: Gram min/max >= {gates['normalized_gram_validity']['threshold']} "
                            "and valid diagonal/retained metric; why: no PSD repair")
                if batch_results is None:
                    model, zero = apply_shared_pole_zero_policy(model, gates=gates)
                    if ordered:
                        zero["zero_policy"] = zero["zero_policy"] & reduction["infinite_weight_ok"]
                if not bool(jnp.all(zero["zero_policy"])):
                    raise ValueError(f"GATE shared_pole_zero_ritz: got: failed at q={q}; want: finite positive response within dropped-weight budget; why: no pole clipping")
                if batch_results is None and ordered:
                    retained = ordered_retained
                elif batch_results is None:
                    # E selects the last infinity block of X. Build it as a face array;
                    # only the small row/column coordinate vectors are replicated.
                    r, ri = pencil[0].shape[-1], qi.shape[-1]
                    selector = jax.jit(lambda: (jnp.arange(r)[:, None] ==
                                               jnp.arange(r-ri, r)[None, :])[None].astype(jnp.complex128),
                                       out_shardings=face)()
                    retained = retained_moment_identity(pencil, coefficients, model, selector, matmul=mm)
                # The ordered identity is on the ORIGINAL infinity directions: exact only for the
                # full Galerkin span, projection accuracy after the keep/retention cuts. It is
                # reported beside the full_m1/full_m3 diagnostic bands, as the TRS route reports
                # its original-direction defects; the retained Ritz algebra is the metric gate above.
                if not ordered and not all(bool(jnp.all(value <= gates["retained_subspace_moments"]["threshold"]))
                                           for value in retained.values()):
                    raise ValueError(f"GATE shared_pole_retained_moments: got: failed at q={q}; want: projected latent moment identity <=1e-10; why: corrected Ritz algebra")
                if batch_results is None and not ordered:
                    del pencil, coefficients, selector
                elif ordered:
                    del pencil
                r = model[0].shape[-1]
                del active_columns
                capacity(model[0].shape[-1], phase="model")
            timing.fence("spole.sort")
            with timing.section("spole.sort"):
                model, permutation = sort_shared_pole_columns(model, mesh_xy=mesh_xy)
            if batch_checks is None:
                timing.fence("spole.coulomb")
                with timing.section("spole.coulomb"):
                    query_workspace("gemm", ((1, n, n), (1, n, n)), eig)
                    capacity(current_side)
                    expose_live((*model, qi))
                    coulomb_sqrt, inverse_sqrt, coulomb_receipt = response_coulomb_powers(
                        meta, config, mesh_xy=mesh_xy, bank_io=bank, q_span=span)
                timing.fence("spole.passivity")
                with timing.section("spole.passivity"):
                    passive = (signed_shared_pole_passivity(
                                   signed, inverse_sqrt, eta_ry=recipe["eta_ev"] / RYD_TO_EV,
                                   matmul=mm, eigh=eig.batched, gates=gates) if ordered else
                               shared_pole_passivity(model, inverse_sqrt,
                                                   eta_ry=recipe["eta_ev"] / RYD_TO_EV,
                                                   matmul=mm, eigh=eig.batched, gates=gates))
                    del coulomb_sqrt, inverse_sqrt
            else:
                timing.fence("spole.passivity")
                with timing.section("spole.passivity"):
                    passive = {key: value[slot:slot+1] for key, value in batch_checks[0].items()}
                    coulomb_receipt = dict(batch_coulomb)
                    coulomb_receipt["support_ranks"] = batch_coulomb["support_ranks"][slot:slot+1]
            timing.fence("spole.gates")
            with timing.section("spole.gates"):
                if not bool(jnp.all(passive["passivity"])):
                    raise ValueError(f"GATE shared_pole_passivity: got: failed at q={q}; want: 0 <= V-whitened -W(i eta) <= I; why: passive screening")
                expose_live((*model, qi))
            timing.fence("spole.moment_read")
            with timing.section("spole.moment_read"):
                with SlabIO(moments["path"], mode="r", mesh=mesh_xy) as moment_io:
                    exact = read_shared_pole_bank(moment_io, span, meta=meta,
                                                  header=moment_header, fields=("M1", "M3"))
            timing.fence("spole.moment_diagnostics")
            with timing.section("spole.moment_diagnostics"):
                if ordered:
                    # Signed model moments: M1 = sum c c^H mu^-2 / 2, M3 = sum c c^H mu^-4 / 2.
                    c_signed, mu_signed, kept = signed
                    inverse = jnp.where(kept, 1 / jnp.where(kept, jnp.abs(mu_signed), 1), 0)
                    moment_defects = model_diagnostics(
                        (c_signed * inverse[:, None, :], inverse**2, kept), exact, qi)
                    del c_signed, mu_signed, kept, inverse
                else:
                    moment_defects = model_diagnostics(model, exact, qi)
                del exact, qi
            timing.fence("spole.held")
            with timing.section("spole.held"):
                if batch_checks is None:
                    held = []
                    reciprocity = []
                    with SlabIO(bank["path"], mode="r", mesh=mesh_xy) as bank_io:
                        for sample_id in recipe["held_ids"]:
                            expose_live(model)
                            samples = read_shared_pole_bank(bank_io, span, meta=meta, header=header,
                                                           sample_span=(int(sample_id), int(sample_id)+1),
                                                           fields=("Wc", "dWc_ds"))
                            b, poles, mask = model
                            s = _sample_point(recipe, int(sample_id)) ** 2
                            weights = jnp.where(mask, 1 / (s-poles), 0)
                            if ordered:
                                # Signed particle-hole model at z; dW/ds = (dW/dz)/(2z).
                                b, mu_signed, mask = signed
                                z = _sample_point(recipe, int(sample_id))
                                weights = jnp.where(mask, 1 / (z * mu_signed - 1), 0)
                                derivative = jnp.where(mask, -mu_signed / (z * mu_signed - 1)**2 / (2 * z), 0)
                            diagnostic = {"sample_id": int(sample_id)}
                            for field, weight in (("Wc", weights), ("dWc_ds", derivative if ordered else -weights**2)):
                                sample = samples[field][:, 0]
                                value = mm(b * weight[:, None, :], b, transb="C")
                                diagnostic[field] = float(jnp.linalg.norm(value-sample) /
                                                          jnp.maximum(jnp.linalg.norm(sample), jnp.finfo(jnp.float64).tiny))
                                if not ordered:
                                    reciprocity.append(shared_pole_reciprocity(value, sample, gates=gates))
                            held.append(diagnostic)
                            del samples, sample, value
                    reciprocity = ({key: np.asarray([row[key] for row in reciprocity]).tolist()
                                    for key in reciprocity[0]} if not ordered else {})
                else:
                    held = [{"sample_id": sample_id,
                             "Wc": float(batch_checks[1][slot, 0, i]),
                             "dWc_ds": float(batch_checks[1][slot, 1, i])}
                            for i, sample_id in enumerate(held_ids)]
                    reciprocity = {key: value[slot].tolist() for key, value in batch_checks[2].items()}
                if not ordered and not np.all(reciprocity["passed"]):
                    raise ValueError(f"GATE shared_pole_model_reciprocity: got: {reciprocity} at q={q}; want: model preserves transpose symmetry of symmetric held data; why: conjugate-port closure must survive reduction")
            timing.fence("spole.receipts")
            with timing.section("spole.receipts"):
                if ordered:
                    del signed
                b, poles, mask = model
                counts = jnp.sum(mask, axis=-1, dtype=jnp.int64)
                # All scalar reductions precede rank-selective store formatting.
                price = capacity(r)
                row = {"q_span": list(span), "roles": roles,
                       "diagnostic_operator": "raw-latent-pole-model",
                       "K": np.asarray(counts).tolist(), "J": int(np.unique(np.asarray(poles)[np.asarray(mask)]).size),
                       "damping_fraction": 0.0, "capacity": price, "coulomb": coulomb_receipt,
                       "condition": np.asarray(reduction["gram_condition"]).tolist(),
                       "normalized_gram_spectrum": np.asarray(reduction["gram_spectrum_relative"])[..., :int(reduction.get("pencil_side", [r])[0])].tolist(),
                       "native_workspace_queries": [dict(op=op, shapes=shapes, bytes_per_rank=value)
                                                     for (op, shapes), value in native_queries.items()],
                       "retained_moment_relative": {k: np.asarray(v).tolist() for k, v in retained.items()},
                       "moment_defects": {k: {a: np.asarray(value).tolist() for a, value in v.items()}
                                          for k, v in moment_defects.items()},
                       "held_W": held, "permutation": np.asarray(permutation).tolist(),
                       "storage_bytes": int(counts[0]) * (16*logical_n + 8)}
                if ordered:
                    row["ordered"] = {key: np.asarray(reduction[key]).tolist() for key in (
                        "positive_count", "negative_count", "infinite_weight_fraction",
                        "paired_rank", "paired_min_relative")}
                    row["ordered"]["odd_moments"] = odd_moments
                row["metric_inverse_root"] = {
                    name: np.asarray(reduction[name]).tolist() for name in (
                        "metric_initial_infinity_norm", "metric_inverse_root_iterations",
                        "metric_inverse_root_residual_fro", "metric_inverse_root_residual_relative")}
                measurements = {
                    "normalized_gram_keep": dict(value=int(reduction["retained_rank"][0]), passed=True, reason="normalized Gram cut, current q"),
                    "normalized_gram_validity": dict(value=float(reduction["gram_min_relative"][0]), passed=True, reason="normalized Gram spectrum"),
                    "zero_ritz_policy": dict(value=float(zero["dropped_factor_weight_fraction"][0]), passed=True, reason="physical factor weight, sentinels excluded"),
                    "finite_factors_poles": dict(value=True, passed=True, reason="zero policy, active prefix and exact inert sentinels"),
                    "passivity": dict(value={k: np.asarray(v).tolist() for k, v in passive.items() if k != "passivity"}, passed=True, reason=("signed particle-hole model, Hermitian part at i eta; anti-Hermitian part is the odd channel, reported" if ordered else "raw latent model; authenticated inverse Coulomb square root at current eta; projected operator not measured")),
                    "retained_subspace_moments": dict(value=row["retained_moment_relative"], passed=True, reason=(("signed model z-moments m0..m3 on the original infinity directions, each order against its own norm; projection-accuracy diagnostic beside full_m1/full_m3, not a refusal" if odd_moments else "finite-state ordered bank without odd moments: infinity block uncertified") if ordered else "raw latent Ritz identity: A=Y†GE, B=YA; pencil B†(G,H)B/2 versus model A†(I,Lambda)A/2")),
                    "held_w": dict(value=held, passed=True, reason="raw latent W and dW/ds diagnostics; projected operator not measured; no universal acceptance threshold"),
                    "model_reciprocity": (dict(value=None, passed=None, reason="not applicable: time-reversal-broken samples carry no transpose symmetry") if ordered else dict(value=reciprocity, passed=True, reason="raw latent model sampled W/dW transpose symmetry, conditional on symmetric reference; projected operator not measured; applicability recorded per sample")),
                    "full_m1_defect": dict(value=float(moment_defects["M1"]["full_relative"][0]), passed=bool(moment_defects["M1"]["full_relative"][0] <= gates["full_m1_defect"]["threshold"]), reason="raw latent model versus physical full M1; projected moment not measured; CD8 diagnostic band, never a refusal"),
                    "full_m3_defect": dict(value=float(moment_defects["M3"]["full_relative"][0]), passed=bool(moment_defects["M3"]["full_relative"][0] <= gates["full_m3_defect"]["threshold"]), reason="raw latent model versus physical full M3; projected moment not measured; CD8 diagnostic band, never a refusal"),
                    "representation": dict(value=({"nspinor": int(meta.nspinor), "trs_allowed": False, "ordered": True} if ordered else {"nspinor": int(meta.nspinor), "trs_allowed": True}), passed=True, reason="current typed symmetry capability"),
                    "capacity": dict(value=price, passed=True, reason="conservative aggregate constructor live-set price"),
                    "sc_rebuild": dict(value=identity, passed=True, reason="current recipe/census authenticated; directions and Ritz model rebuilt"),
                }
                receipt = construction_receipt(
                    measurements, capacity=ledger,
                    capacity_entry_start=receipt_entry_start, ordered=ordered)
                receipt_entry_start = len(ledger.entries)
                receipt.update(identity=identity, constructor=row)
            timing.fence("spole.export_prepare")
            with timing.section("spole.export_prepare"):
                public_b = _public_factor_kernel(mesh_xy)(b)
                del model, b, mask
                ready_models.append((public_b, poles, counts))
                retained_panels = (*retained_panels, public_b, poles, counts)
                receipts.append(receipt)
                del public_b, poles, counts
        timing.fence("spole.reduction_admission")
        with timing.section("spole.reduction_admission"):
            batch_width = len(selected)
            capacity(ready_models[0][0].shape[-1], phase="model")
        timing.fence("spole.writer_stack")
        with timing.section("spole.writer_stack"):
            public_b, poles, counts = stack_models(tuple(ready_models))
            expose_live((public_b, poles, counts))
            span = (selected[0][0], selected[-1][0] + 1)
            batch_receipt = {"identity": identity,
                             "q_receipts": receipts[-len(selected):]}
        timing.fence("spole.writer")
        with timing.section("spole.writer"):
            store_header = write_shared_pole_model(
                output, public_b, poles, counts, q_span=span, meta=meta,
                tables=bank["tables"], recipe=recipe, receipts=batch_receipt, ordered=ordered)
        timing.fence("spole.cleanup")
        with timing.section("spole.cleanup"):
            del public_b, poles, counts, ready_models
            ledger.live_stages = upstream
            del selected, batch_results
            retained_panels = ()
    timing.fence("spole.return")
    with timing.section("spole.return"):
        return {"q_receipts": receipts, "model_header": store_header,
                "capacity": ledger.receipt(),
                "identity": identity, "status": "CONSTRUCTED"}

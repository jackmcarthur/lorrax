"""Frozen-static screened finite-q crossing block.

Production constructs spectral-projector moments from the centroid-space
resolvent.  Every contour node evaluates the landed WP1 crossing kernel and
solves only an ``N_mu x N_mu`` system.  Pair data remain the two sharded
``N_s x N_mu`` vertex tables plus ``O(N_s)`` scalars; production never forms
``H`` or any other ``N_s x N_s`` object.

The old pair-space eigensolve survives only as
:func:`_dense_reference_modes`, a tests-only oracle for small fixtures.  The
production moments are compressed into the existing pole-store convention;
the store and Sigma lifecycle are deliberately unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P


MODEL = "intraband_eigenmode_v1"

# Numerical acceptances, never input keys or physical parameters.
SAMPLE_REL_TOL = 4.0e-3
STATIC_REL_TOL = 2.0e-11
SUM_RULE_REL_TOL = 1.0e-12
MOMENT_REL_TOL = SAMPLE_REL_TOL
# A static deficit is admitted as zero-mode weight only when doubling the
# quadrature order leaves it this stationary; anything that still shrinks is
# quadrature error and keeps refining toward the sum-rule refusal.
_V_PLATEAU_REL_TOL = 1.0e-3
MIN_CLUSTERS = 3
MAX_CLUSTERS = 6
_QUADRATURE_ORDERS = (16, 32, 64, 128, 256, 512)
_RESOLVENT_BATCH_NODES = 8

_DENSE_REFERENCE_KERNELS = {}
_RESOLVENT_PLANS = {}


@dataclass(frozen=True)
class IntrabandRow:
    """One compressed wedge row, still in native distributed sharding."""

    Omega_p: jax.Array
    B_p: jax.Array
    n_poles: int
    n_modes: int
    sample_max_rel_error: float
    static_max_rel_error: float
    certified: bool
    folded_modes: int
    dropped_modes: int
    folded_elements: int
    dropped_elements: int
    cluster_width_max_ry: float
    zero_mode_weight: float = 0.0
    zero_mode_cluster: int = -1
    zero_mode_pole_shift: float = 0.0


@dataclass(frozen=True)
class ClusterClosure:
    """Post-merge closure record for one row's contour tiling.

    ``zero_mode_weight`` is ``||D_V||/||V_total||`` measured *before* the
    deflation merge; after the merge both closures are exact identities.
    """

    quadrature_order: int
    m_closure: float
    v_closure_before_merge: float
    v_closure_after_merge: float
    zero_mode_weight: float
    zero_mode_cluster: int
    zero_mode_pole_shift: float


def _mesh_of(value, where):
    mesh = getattr(getattr(value, "sharding", None), "mesh", None)
    if mesh is None:
        raise ValueError(f"{where} requires a globally NamedSharded array")
    return mesh


def _host_replicated(value):
    """Host view of a deliberately replicated small scalar table."""
    return np.asarray(jax.device_get(value.addressable_data(0)))


def _dense_reference_modes(W0bar, pair_block):
    """Return dense pair-space modes for tests; never called by production.

    This is the claim-0315 eigensolve retained as the <=2k synthetic oracle.
    Its private name and isolated cache make an accidental production call
    visible in review and in the no-``N_s**2`` acceptance test.
    """
    mesh = _mesh_of(W0bar, "_dense_reference_modes")
    kernel = _DENSE_REFERENCE_KERNELS.get(id(mesh))
    if kernel is None:
        rep1 = NamedSharding(mesh, P(None))
        left_shard = NamedSharding(mesh, P("x", None))
        right_shard = NamedSharding(mesh, P(None, "y"))

        @jax.jit
        def kernel(W0, u, w, p_x, p_y):
            Wp = jax.lax.with_sharding_constraint(
                W0 @ jnp.transpose(p_y), left_shard)
            pW = jax.lax.with_sharding_constraint(
                jnp.conj(p_x) @ W0, right_shard)
            A = jax.lax.with_sharding_constraint(
                jnp.conj(p_x) @ Wp,
                NamedSharding(mesh, P(None, None)))
            H = jnp.diag(u * u) + (2.0 * w[:, None]) * A
            eigenvalues, X = jnp.linalg.eig(H)
            Xinv = jnp.linalg.inv(X)
            left = jax.lax.with_sharding_constraint(Wp @ X, left_shard)
            right = jax.lax.with_sharding_constraint(
                Xinv @ ((-2.0 * w)[:, None] * pW), right_shard)
            left_norm = jnp.sqrt(jnp.sum(jnp.abs(left) ** 2, axis=0))
            right_norm = jnp.sqrt(jnp.sum(jnp.abs(right) ** 2, axis=1))
            weights = jax.lax.with_sharding_constraint(
                left_norm * right_norm, rep1)
            return eigenvalues, left, right, weights

        _DENSE_REFERENCE_KERNELS[id(mesh)] = kernel

    u, w, vertices = pair_block
    p_x, p_y = vertices
    return kernel(W0bar, u, w, p_x, p_y)


def _z_for_zeta(zeta):
    """Return a square root accepted by WP1; only its square enters chi1."""
    root = np.sqrt(complex(zeta))
    if root.imag < 0.0:
        root = -root
    return complex(root)


def _face_matmul(A, B, mesh):
    """Multiply face-sharded matrices without a replicated production tile."""
    if jax.process_count() == 1:
        return A @ B
    from ffi import _services
    _services.ensure_on_path()
    from distrib_la import matmul
    return matmul(A, B, mesh=mesh, backend="distributed")


def _resolvents_at_zeta(pair_block, W0bar, zetas):
    """Evaluate a small node batch with face-sharded GEMMs and N_mu solves."""
    from gw.w_isdf import intraband_chi1

    zetas = tuple(complex(value) for value in zetas)
    if not zetas:
        raise ValueError("_resolvents_at_zeta requires at least one node")
    mesh = _mesh_of(W0bar, "_resolvents_at_zeta")
    extent = int(W0bar.shape[0])
    matrix_shard = NamedSharding(mesh, P("x", "y"))
    stack_shard = NamedSharding(mesh, P(None, "x", "y"))
    chi_stack = jax.lax.with_sharding_constraint(
        jnp.stack([
            intraband_chi1(pair_block, _z_for_zeta(zeta))
            for zeta in zetas
        ]),
        stack_shard,
    )
    W_stack = jax.lax.with_sharding_constraint(
        jnp.broadcast_to(W0bar, (len(zetas), extent, extent)), stack_shard)
    Wchi = jax.lax.with_sharding_constraint(
        _face_matmul(W_stack, chi_stack, mesh), stack_shard)
    rhs = jax.lax.with_sharding_constraint(
        _face_matmul(Wchi, W_stack, mesh), stack_shard)
    identity = jnp.eye(extent, dtype=jnp.complex128)[None, :, :]
    system = jax.lax.with_sharding_constraint(identity - Wchi, stack_shard)

    if jax.process_count() == 1:
        result = jnp.linalg.solve(system, rhs)
    else:
        key = (id(mesh), extent)
        solve_plan = _RESOLVENT_PLANS.get(key)
        if solve_plan is None:
            from ffi import _services
            _services.ensure_on_path()
            from distrib_la import plan as linalg_plan
            solve_plan = linalg_plan(
                "solve_lu", mesh, backend="distributed", n=extent)
            _RESOLVENT_PLANS[key] = solve_plan
            if jax.process_index() == 0:
                print(
                    "  [intraband-contour] resolvent_solve="
                    f"{solve_plan.describe()} batch_nodes="
                    f"{_RESOLVENT_BATCH_NODES}",
                    flush=True,
                )
        # solve_lu donates both operands; system and rhs are fresh buffers.
        result = solve_plan.batched(system, rhs)
    return jax.lax.with_sharding_constraint(result, stack_shard)


def _resolvent_at_zeta(pair_block, W0bar, zeta):
    """Direct one-node evaluation, including the independent zeta=0 anchor."""
    mesh = _mesh_of(W0bar, "_resolvent_at_zeta")
    matrix_shard = NamedSharding(mesh, P("x", "y"))
    return jax.lax.with_sharding_constraint(
        _resolvents_at_zeta(pair_block, W0bar, (zeta,))[0], matrix_shard)


def _exact_moment_totals(pair_block, W0bar):
    """Independent static and high-frequency anchors for contour closure."""
    _u, w, vertices = pair_block
    p_x, p_y = vertices
    mesh = _mesh_of(W0bar, "_exact_moment_totals")
    matrix_shard = NamedSharding(mesh, P("x", "y"))
    V_total = _resolvent_at_zeta(pair_block, W0bar, 0.0j)
    C1 = jax.lax.with_sharding_constraint(
        jnp.einsum(
            "s,sm,sn->mn", 2.0 * w, p_x, jnp.conj(p_y), optimize=True),
        matrix_shard,
    )
    M_total = jax.lax.with_sharding_constraint(
        _face_matmul(_face_matmul(W0bar, C1, mesh), W0bar, mesh),
        matrix_shard)
    return M_total, V_total


def _relative_error(model, exact):
    numerator = jnp.real(jnp.vdot(model - exact, model - exact))
    denominator = jnp.real(jnp.vdot(exact, exact))
    floor = np.finfo(np.float64).tiny
    return float(jax.device_get(jnp.sqrt(
        numerator / jnp.maximum(denominator, floor))))


def _frobenius_norm(value):
    return float(jax.device_get(jnp.sqrt(
        jnp.real(jnp.vdot(value, value)))))


def _contour_geometry(pair_block, W0bar):
    """Derived two-sided zeta strip, with the zeta=0 pole excluded."""
    u, w, _vertices = pair_block
    u_host = _host_replicated(u).astype(np.float64, copy=False)
    w_host = _host_replicated(w).astype(np.float64, copy=False)
    lambda_q = float(np.max(np.abs(u_host)))
    # Frobenius is a valid upper bound for the unspecified matrix norm in the
    # ruling and avoids a second spectral problem at N_mu scale.
    w_norm = float(jax.device_get(jnp.linalg.norm(W0bar)))
    interaction_bound = w_norm * float(np.sum(2.0 * np.abs(w_host)))
    bare_zeta = lambda_q * lambda_q
    interaction = interaction_bound
    zeta_max = bare_zeta + interaction
    if not np.isfinite(zeta_max) or zeta_max <= 0.0:
        raise ValueError(
            "GATE intraband_contour_domain: derived zeta_max must be finite "
            f"and positive; got {zeta_max!r}")
    zeta_max = float(np.nextafter(zeta_max, np.inf))
    positive_u2 = np.square(np.abs(u_host[u_host != 0.0]))
    if positive_u2.size == 0:
        raise ValueError(
            "GATE intraband_contour_domain: crossing block has no nonzero "
            "transition energy")
    # H=diag(u**2)+low-rank is non-Hermitian when MP1 weights change sign.
    # The norm bound therefore applies on *both* sides of the bare positive
    # spectrum.  Production row 2 is the decisive case: a positive-only
    # strip missed half of both exact totals.  That is a mandatory sum-rule
    # refusal, not a physical model.
    zeta_min = -interaction
    # Zero must remain outside every V contour.  Keeping both inner edges a
    # resolved distance from sqrt's branch point is required for T1
    # quadrature convergence; the independent M/V closures refuse if a mode
    # falls into this machine-only gap.
    zero_gap = (
        np.finfo(np.float64).eps * zeta_max
        if np.any(w_host < 0.0)
        else max(0.25 * float(np.min(positive_u2)),
                 np.finfo(np.float64).eps * zeta_max)
    )
    # The interaction radius is the derived imaginary excursion.  Its only
    # floor is roundoff separation from the real axis.
    height = max(interaction,
                 128.0 * np.finfo(np.float64).eps * zeta_max)
    if not 0.0 < zero_gap < zeta_max:
        raise ValueError(
            "GATE intraband_contour_domain: derived two-sided strip cannot "
            "exclude zeta=0: "
            f"zeta_min={zeta_min:.17e}, zero_gap={zero_gap:.17e}, "
            f"zeta_max={zeta_max:.17e}")
    return zeta_min, zero_gap, zeta_max, height


def _initial_intervals(pair_block, W0bar):
    """Three certified tiles, extending negative for signed MP1 weights."""
    left, zero_gap, right, _height = _contour_geometry(pair_block, W0bar)
    u2 = np.sort(np.square(np.abs(_host_replicated(pair_block[0]))))
    w = _host_replicated(pair_block[1])
    negative_active = bool(np.any(w < 0.0) and left < -zero_gap)
    n_positive = MIN_CLUSTERS - int(negative_active)
    edges = [zero_gap]
    gap_order = np.argsort(np.diff(u2))[::-1][:n_positive - 1]
    for gap_index in np.sort(gap_order):
        edge = 0.5 * (float(u2[gap_index]) + float(u2[gap_index + 1]))
        edge = min(max(edge, np.nextafter(edges[-1], np.inf)), right)
        edges.append(edge)
    edges.append(right)
    if any(not lo < hi for lo, hi in zip(edges[:-1], edges[1:])):
        edges = list(np.linspace(zero_gap, right, n_positive + 1))
    intervals = ([(float(left), float(-zero_gap))]
                 if negative_active else [])
    intervals.extend((float(lo), float(hi))
                     for lo, hi in zip(edges[:-1], edges[1:]))
    return intervals


def _quadrature_nodes(interval, height, order):
    """Counter-clockwise Gauss-Legendre nodes and dz weights on a rectangle."""
    lo, hi = (float(interval[0]), float(interval[1]))
    x, weight = np.polynomial.legendre.leggauss(int(order))
    coarse_segments = (
        (complex(lo, -height), complex(hi, -height)),
        (complex(hi, -height), complex(hi, height)),
        (complex(hi, height), complex(lo, height)),
        (complex(lo, height), complex(lo, -height)),
    )
    segments = []
    for start, stop in coarse_segments:
        if start.real == stop.real and start.imag * stop.imag < 0.0:
            scale = min(
                height,
                max(abs(start.real),
                    np.finfo(np.float64).eps * height),
            )
            ordinates = [0.0]
            radius = scale
            while radius < height:
                ordinates.extend((-radius, radius))
                radius *= 2.0
            ordinates.extend((-height, height))
            ordinates = sorted(set(ordinates))
            if start.imag > stop.imag:
                ordinates.reverse()
            points = [complex(start.real, value) for value in ordinates]
            segments.extend(zip(points[:-1], points[1:]))
        else:
            segments.append((start, stop))
    for start, stop in segments:
        midpoint = 0.5 * (start + stop)
        half = 0.5 * (stop - start)
        for xi, wi in zip(x, weight):
            yield midpoint + half * xi, half * wi


def _retarded_sqrt_on_interval(zeta, interval):
    """Analytic square-root branch local to one side of the zero gap."""
    if float(interval[1]) < 0.0:
        # sqrt(-zeta) has its cut on the positive-zeta axis, outside a
        # negative interval contour.  The -i sheet is the retarded fold.
        return -1.0j * np.sqrt(-complex(zeta))
    return np.sqrt(complex(zeta))


def _moments_at_order(
        pair_block, W0bar, intervals, order, V_total, *, height=None):
    mesh = _mesh_of(W0bar, "_moments_at_order")
    matrix_shard = NamedSharding(mesh, P("x", "y"))
    pole_shard = NamedSharding(mesh, P(None, "x", "y"))
    _left, _zero_gap, _right, max_height = _contour_geometry(
        pair_block, W0bar)
    height = max_height if height is None else float(height)
    shape = tuple(W0bar.shape)
    M_rows, V_rows, T1_rows, T2_rows = [], [], [], []
    normalization = 1.0 / (2.0j * np.pi)
    for interval in intervals:
        M = jnp.zeros(shape, dtype=jnp.complex128)
        V = jnp.zeros(shape, dtype=jnp.complex128)
        T1 = jnp.zeros(shape, dtype=jnp.complex128)
        T2 = jnp.zeros(shape, dtype=jnp.complex128)
        nodes = tuple(_quadrature_nodes(interval, height, order))
        for start in range(0, len(nodes), _RESOLVENT_BATCH_NODES):
            batch = nodes[start:start + _RESOLVENT_BATCH_NODES]
            R_batch = _resolvents_at_zeta(
                pair_block, W0bar, (zeta for zeta, _dz in batch))
            # A globally sharded jax.Array deliberately refuses Python's
            # iterator protocol: no process owns the full face.  The node
            # coordinate is replicated, so integer indexing is the legal
            # global operation and leaves the matrix at P('x','y').
            for node_index, (zeta, dz) in enumerate(batch):
                R = R_batch[node_index]
                factor = normalization * dz
                M = M + factor * R
                # R=C/(lambda-zeta), so the CCW residue of R/zeta is
                # -C/lambda.  This minus makes V=R(0).  Subtracting R(0)
                # analytically removes the nearby but excluded zeta=0 pole.
                V = V - factor * (R - V_total) / zeta
                T1 = T1 + factor * _retarded_sqrt_on_interval(
                    zeta, interval) * R
                T2 = T2 + factor * zeta * R
        M_rows.append(jax.lax.with_sharding_constraint(M, matrix_shard))
        V_rows.append(jax.lax.with_sharding_constraint(V, matrix_shard))
        T1_rows.append(jax.lax.with_sharding_constraint(T1, matrix_shard))
        T2_rows.append(jax.lax.with_sharding_constraint(T2, matrix_shard))
    return tuple(
        jax.lax.with_sharding_constraint(jnp.stack(rows), pole_shard)
        for rows in (M_rows, V_rows, T1_rows, T2_rows)
    )


def _representative_omega(M, V):
    """Trace-moment pole location per cluster, on the retarded sheet.

    This is the same trace-ratio idiom the intrinsic widths already use.  It
    is a diagnostic summary of the elementwise match ``Omega^2 = -M/V``, and
    it is what "lowest-|Omega| cluster" means as a fixed structural rule.
    """
    roots = []
    for index in range(int(M.shape[0])):
        mass = complex(jax.device_get(jnp.trace(M[index])))
        static = complex(jax.device_get(jnp.trace(V[index])))
        if static == 0.0:
            roots.append(complex(np.inf, 0.0))
            continue
        ratio = -mass / static
        if not np.isfinite(ratio):
            roots.append(complex(np.inf, 0.0))
            continue
        root = np.sqrt(complex(ratio))
        if root.real < 0.0:
            root = -root
        if root.imag > 0.0:
            root = root.conjugate()
        roots.append(complex(root))
    return roots


def _finite_lambda_clusters(M, M_total):
    """Clusters carrying material asymptotic weight, i.e. finite-lambda.

    A machine-zero mode has ``M``-content proportional to its own ``lambda``,
    so a cluster with no asymptotic weight cannot host the deflated static
    weight; §2.4b's merge target must be a genuine finite-lambda cluster.
    """
    scale = max(_frobenius_norm(M_total), np.finfo(np.float64).tiny)
    return [
        index for index in range(int(M.shape[0]))
        if _frobenius_norm(M[index]) / scale > SUM_RULE_REL_TOL
    ]


def _merge_zero_mode(M, V, M_total, V_total, m_closure, v_closure, order):
    """§2.4b deflation-merge, applied before the two-moment match.

    The dichotomy is codified here and nowhere else: an open ``M`` closure is
    a missed *finite*-lambda mode and refuses; a closed ``M`` closure with a
    nonzero ``D_V`` is the machine-zero screened mode's static weight, which
    the sum rule defines with no free parameter.
    """
    if m_closure > SUM_RULE_REL_TOL:
        raise ValueError(
            "GATE intraband_contour_sum_rule: the asymptotic closure is open "
            f"(||D_M||/||M_total||={m_closure:.6e} > "
            f"{SUM_RULE_REL_TOL:.1e}) at quadrature order {order}.  A missed "
            "finite-lambda mode carries asymptotic weight, so it can never "
            "masquerade as zero-mode static weight; the static deficiency "
            f"(||D_V||/||V_total||={v_closure:.6e}) is NOT merged.  Fix the "
            "contour tiling.")

    D_V = V_total - jnp.sum(V, axis=0)
    candidates = _finite_lambda_clusters(M, M_total)
    if not candidates:
        raise ValueError(
            "GATE intraband_zero_mode_merge: no finite-lambda cluster exists "
            "to receive the deflated zero-mode static weight "
            f"(||D_V||/||V_total||={v_closure:.6e}); every contour is "
            "asymptotically dark.")
    before = _representative_omega(M, V)
    target = min(candidates, key=lambda index: abs(before[index]))

    index_axis = jnp.arange(int(V.shape[0]))[:, None, None]
    merged = jnp.where(index_axis == target, V + D_V[None, :, :], V)
    merged = jax.lax.with_sharding_constraint(merged, V.sharding)
    after = _representative_omega(M, merged)
    reference = max(abs(before[target]), np.finfo(np.float64).tiny)
    pole_shift = float(abs(after[target] - before[target]) / reference)

    v_after = _relative_error(jnp.sum(merged, axis=0), V_total)
    m_after = _relative_error(jnp.sum(M, axis=0), M_total)
    if jax.process_index() == 0:
        print(
            "[intraband-zero-mode] "
            f"merge_cluster={target} n_clusters={int(V.shape[0])} "
            f"zero_mode_weight={v_closure:.6e} "
            f"omega_c1_before={abs(before[target]):.6e} "
            f"omega_c1_after={abs(after[target]):.6e} "
            f"pole_shift_rel={pole_shift:.6e} "
            f"post_merge_Sigma_M_rel={m_after:.6e} "
            f"post_merge_Sigma_V_rel={v_after:.6e}",
            flush=True,
        )
    if m_after > SUM_RULE_REL_TOL or v_after > SUM_RULE_REL_TOL:
        raise ValueError(
            "GATE intraband_contour_sum_rule: post-merge closure is not an "
            f"identity: Sigma_M_rel={m_after:.6e}, Sigma_V_rel={v_after:.6e}, "
            f"tolerance={SUM_RULE_REL_TOL:.1e}")
    return merged, ClusterClosure(
        quadrature_order=int(order),
        m_closure=float(m_closure),
        v_closure_before_merge=float(v_closure),
        v_closure_after_merge=float(v_after),
        zero_mode_weight=float(v_closure),
        zero_mode_cluster=int(target),
        zero_mode_pole_shift=pole_shift,
    )


def _cluster_moment_matrices(
        pair_block, W0bar, intervals, *, moment_rel_tol=MOMENT_REL_TOL):
    """Contour M/V/T1/T2 with mandatory movement and sum-rule refusals.

    Returns ``(M, V, T1, T2, closure)``.  ``V`` already carries the §2.4b
    deflation merge, so both sum rules hold as exact identities on the
    returned moments.
    """
    intervals = tuple((float(lo), float(hi)) for lo, hi in intervals)
    if not intervals or any(not lo < hi for lo, hi in intervals):
        raise ValueError(
            "GATE intraband_contour_intervals: intervals must be nonempty "
            "ordered (lo,hi) pairs")
    left, zero_gap, right, height = _contour_geometry(pair_block, W0bar)
    negative = tuple(value for value in intervals if value[1] < 0.0)
    positive = tuple(value for value in intervals if value[0] > 0.0)
    negative_domain_ok = (
        (negative
         and abs(negative[0][0] - left) <= 8.0 * abs(np.spacing(left))
         and abs(negative[-1][1] + zero_gap)
         <= 8.0 * abs(np.spacing(zero_gap)))
        or not negative
    )
    tiled = (
        len(negative) + len(positive) == len(intervals)
        and positive
        and negative_domain_ok
        and abs(positive[0][0] - zero_gap)
        <= 8.0 * abs(np.spacing(zero_gap))
        and abs(positive[-1][1] - right) <= 8.0 * abs(np.spacing(right))
        and all(a[1] == b[0]
                for group in (negative, positive)
                for a, b in zip(group[:-1], group[1:]))
        and tuple((*negative, *positive)) == intervals
    )
    if not tiled:
        raise ValueError(
            "GATE intraband_contour_intervals: intervals do not exactly tile "
            "a closure-certifiable strip "
            f"([{left:.17e},{-zero_gap:.17e}] U) "
            f"[{zero_gap:.17e},{right:.17e}]")

    M_total, V_total = _exact_moment_totals(pair_block, W0bar)
    previous = None
    previous_v_closure = None
    movement = np.inf
    m_closure = np.inf
    v_closure = np.inf
    for order in _QUADRATURE_ORDERS:
        current = _moments_at_order(
            pair_block, W0bar, intervals, order, V_total)
        movements = (np.inf,) * 4
        if previous is not None:
            mass_scale = max(
                sum(_frobenius_norm(current[0][index])
                    for index in range(int(current[0].shape[0]))),
                sum(_frobenius_norm(previous[0][index])
                    for index in range(int(previous[0].shape[0]))),
                np.finfo(np.float64).tiny,
            )
            static_scale = max(
                sum(_frobenius_norm(current[1][index])
                    for index in range(int(current[1].shape[0]))),
                sum(_frobenius_norm(previous[1][index])
                    for index in range(int(previous[1].shape[0]))),
                np.finfo(np.float64).tiny,
            )
            zeta_radius = float(np.hypot(max(abs(left), right), height))
            scales = (
                mass_scale,
                static_scale,
                np.sqrt(zeta_radius) * mass_scale,
                zeta_radius * mass_scale,
            )
            movements = tuple(
                _frobenius_norm(value - old) / scale
                for value, old, scale in zip(current, previous, scales))
            movement = max(movements)
        m_closure = _relative_error(jnp.sum(current[0], axis=0), M_total)
        v_closure = _relative_error(jnp.sum(current[1], axis=0), V_total)
        if jax.process_index() == 0:
            print(
                "[intraband-contour] "
                f"clusters={len(intervals)} order={order} "
                f"movement={movement:.6e} "
                "move_M/V/T1/T2="
                f"{movements[0]:.3e}/{movements[1]:.3e}/"
                f"{movements[2]:.3e}/{movements[3]:.3e} "
                f"Sigma_M_rel={m_closure:.6e} "
                f"Sigma_V_rel={v_closure:.6e}",
                flush=True,
            )
        # A machine-zero screened mode leaves a static deficit that no
        # refinement removes (claim 0319: 3.277321e-3 at orders 16 and 32).
        # Quadrature error does not behave that way, so the static closure
        # stays a convergence criterion until it *plateaus*: only a
        # refinement-invariant D_V is admitted as zero-mode weight, and a
        # shrinking one keeps refining to the order-512 refusal.
        plateaued = (
            previous_v_closure is not None
            and abs(v_closure - previous_v_closure)
            <= _V_PLATEAU_REL_TOL * max(v_closure, previous_v_closure)
        )
        # Once the quadrature has converged, what is left is the spectral
        # domain, not the rule.  Adjudicate the dichotomy here: an open M
        # closure refuses by name now rather than burning six more orders
        # into a message that never says why.
        settled = (m_closure > SUM_RULE_REL_TOL
                   or v_closure <= SUM_RULE_REL_TOL
                   or plateaued)
        if movement <= float(moment_rel_tol) and settled:
            merged, closure = _merge_zero_mode(
                current[0], current[1], M_total, V_total,
                m_closure, v_closure, order)
            return (current[0], merged, current[2], current[3], closure)
        previous = current
        previous_v_closure = v_closure
    raise ValueError(
        "GATE intraband_contour_sum_rule: quadrature failed mandatory "
        f"closure at order {_QUADRATURE_ORDERS[-1]}: "
        f"moment_movement={movement:.6e}, Sigma_M_rel={m_closure:.6e}, "
        f"Sigma_V_rel={v_closure:.6e}, sum_tolerance={SUM_RULE_REL_TOL:.1e}, "
        f"movement_tolerance={float(moment_rel_tol):.1e}")


def _cluster_widths(M, T1, T2):
    """Trace moments -> one parameter-free intrinsic width per interval."""
    widths = []
    for index in range(int(M.shape[0])):
        mass = complex(jax.device_get(jnp.trace(M[index])))
        first = complex(jax.device_get(jnp.trace(T1[index])))
        second = complex(jax.device_get(jnp.trace(T2[index])))
        if mass == 0.0:
            widths.append(0.0)
            continue
        mean1 = np.real(first / mass)
        mean2 = np.real(second / mass)
        variance = float(mean2 - mean1 * mean1)
        tolerance = 256.0 * np.finfo(np.float64).eps * max(abs(mean2), 1.0)
        if variance < -tolerance and mean2 < 0.0:
            # mean zeta < 0 is the overdamped side of the strip: Omega there
            # is purely imaginary, so the real-Omega trace variance is not an
            # intrinsic width and its sign carries no information.  Name it
            # and carry zero rather than refuse on a quantity the retarded
            # fold will not consume (only near-real roots take a width).
            if jax.process_index() == 0:
                print(
                    "[intraband-cluster-width] overdamped interval "
                    f"{index}: mean_zeta={mean2:.6e} < 0, trace variance "
                    f"{variance:.6e} is not a real-Omega width; width=0",
                    flush=True,
                )
            widths.append(0.0)
            continue
        if variance < -tolerance:
            raise ValueError(
                "GATE intraband_cluster_width: material negative trace "
                f"variance {variance:.6e} in interval {index}")
        if abs(variance) <= tolerance:
            variance = 0.0
        widths.append(float(np.sqrt(max(variance, 0.0))))
    return np.asarray(widths, dtype=np.float64)


def _compress_moments(mesh, M, V, T1, T2):
    """The landed elementwise two-moment match fed by contour moments."""
    matrix_shard = NamedSharding(mesh, P("x", "y"))
    pole_shard = NamedSharding(mesh, P(None, "x", "y"))
    # A derived norm bound may leave a contour with provably no enclosed
    # mode.  Its numerical integral is roundoff, not a stored pole.  Removing
    # such a row is allowed only far inside the already-enforced sum-rule
    # tolerance; both independent totals must say it is empty.
    M_scale = max(float(jax.device_get(jnp.linalg.norm(jnp.sum(M, axis=0)))),
                  np.finfo(np.float64).tiny)
    V_scale = max(float(jax.device_get(jnp.linalg.norm(jnp.sum(V, axis=0)))),
                  np.finfo(np.float64).tiny)
    keep = np.asarray([
        (float(jax.device_get(jnp.linalg.norm(M[index]))) / M_scale
         > 0.1 * SUM_RULE_REL_TOL)
        or (float(jax.device_get(jnp.linalg.norm(V[index]))) / V_scale
            > 0.1 * SUM_RULE_REL_TOL)
        for index in range(int(M.shape[0]))
    ], dtype=bool)
    if not np.any(keep):
        raise ValueError(
            "GATE intraband_contour_empty: every certified contour has "
            "zero M and V weight")
    active = jnp.asarray(np.flatnonzero(keep), dtype=jnp.int32)
    M, V, T1, T2 = (value[active] for value in (M, V, T1, T2))
    widths = _cluster_widths(M, T1, T2)
    Omega_rows, B_rows = [], []
    folded_elements = 0
    dropped_elements = 0
    for group, width in enumerate(widths):
        Mc = M[group]
        Vc = V[group]
        scale = jnp.maximum(jnp.max(jnp.abs(Vc)), 1.0)
        live = jnp.abs(Vc) > np.finfo(np.float64).eps * scale
        lambda_bar = jnp.where(live, -Mc / Vc, 1.0 + 0.0j)
        root_raw = jnp.sqrt(lambda_bar)
        root_raw = jnp.where(jnp.real(root_raw) < 0.0, -root_raw, root_raw)
        fold = live & (jnp.imag(root_raw) > 0.0)
        root = jnp.where(fold, jnp.conj(root_raw), root_raw)
        invalid = live & (~jnp.isfinite(root) | (jnp.real(root) <= 0.0))
        active = live & ~invalid
        root_scale = jnp.maximum(jnp.max(jnp.abs(root)), 1.0)
        near_real = jnp.abs(jnp.imag(root)) <= (
            64.0 * np.finfo(np.float64).eps * root_scale)
        root = jnp.where(active & near_real & (width > 0.0),
                         jnp.real(root) - 1.0j * width, root)
        root = jnp.where(active, root, 1.0 + 0.0j)
        # Re-solve from V after a retarded fold or imposed trace width.
        B = jnp.where(active, -Vc * root / 2.0, 0.0 + 0.0j)
        Omega_rows.append(jax.lax.with_sharding_constraint(root, matrix_shard))
        B_rows.append(jax.lax.with_sharding_constraint(B, matrix_shard))
        folded_elements += int(jax.device_get(jnp.count_nonzero(fold)))
        dropped_elements += int(jax.device_get(jnp.count_nonzero(invalid)))
    return (
        jax.lax.with_sharding_constraint(jnp.stack(Omega_rows), pole_shard),
        jax.lax.with_sharding_constraint(jnp.stack(B_rows), pole_shard),
        folded_elements,
        dropped_elements,
        float(np.max(widths, initial=0.0)),
    )


def evaluate_pole_sum(Omega_p, B_p, z):
    """Evaluate poles in the store's ``2 Omega B/(z^2-Omega^2)`` form."""
    zc = jnp.asarray(complex(z), dtype=jnp.complex128)
    return jnp.sum(
        2.0 * Omega_p * B_p / (zc * zc - Omega_p * Omega_p), axis=0)


def _split_largest_trace_interval(intervals, M, pair_block):
    weights = np.asarray([
        abs(complex(jax.device_get(jnp.trace(M[index]))))
        for index in range(len(intervals))
    ])
    u2 = np.square(np.abs(_host_replicated(pair_block[0])))
    members = []
    for index, (lo, hi) in enumerate(intervals):
        values = np.sort(u2[(u2 >= lo) & (u2 <= hi)])
        members.append(values)
        if values.size < 2:
            weights[index] = -np.inf
    if not np.any(np.isfinite(weights)):
        raise ValueError(
            "GATE intraband_cluster_budget: no interval containing two bare "
            "crossing energies remains available for bisection")
    index = int(np.argmax(weights))
    lo, hi = intervals[index]
    values = members[index]
    gaps = np.diff(values)
    gap_index = int(np.argmax(gaps))
    midpoint = 0.5 * (float(values[gap_index])
                      + float(values[gap_index + 1]))
    if not lo < midpoint < hi:
        raise ValueError(
            "GATE intraband_cluster_budget: largest-moment interval cannot "
            f"be bisected: [{lo:.17e},{hi:.17e}]")
    return (intervals[:index]
            + [(lo, midpoint), (midpoint, hi)]
            + intervals[index + 1:])


def _print_memory_model(pair_block, W0bar, n_interval):
    n_pair = int(pair_block[0].shape[0])
    n_mu = int(W0bar.shape[0])
    scalar_bytes = 2 * n_pair * np.dtype(np.float64).itemsize
    vertex_bytes = 2 * n_pair * n_mu * np.dtype(np.complex128).itemsize
    matrix_bytes = n_mu * n_mu * np.dtype(np.complex128).itemsize
    if jax.process_index() == 0:
        print(
            "[intraband-memory] "
            f"n_pair={n_pair} n_mu={n_mu} intervals={n_interval} "
            f"pair_scalars_bytes={scalar_bytes} "
            f"pair_vertices_bytes={vertex_bytes} "
            f"largest_matrix_bytes={matrix_bytes} ns_squared_arrays=0",
            flush=True,
        )


def build_row(W0bar, pair_block, z_samples, *, sample_rel_tol=SAMPLE_REL_TOL):
    """Build one row by certified contour moments and greedy bisection."""
    mesh = _mesh_of(W0bar, "build_row")
    n_pair = int(pair_block[0].shape[0])
    if n_pair == 0:
        raise ValueError("build_row is not called for the empty Gamma block")

    z_values = tuple(
        complex(value) for value in np.asarray(z_samples).reshape(-1))
    intervals = _initial_intervals(pair_block, W0bar)
    selected = None
    while True:
        M, V, T1, T2, closure = _cluster_moment_matrices(
            pair_block, W0bar, intervals,
            moment_rel_tol=min(MOMENT_REL_TOL, float(sample_rel_tol)))
        Om, Bp, fold_el, drop_el, width = _compress_moments(
            mesh, M, V, T1, T2)
        errors = [
            _relative_error(
                evaluate_pole_sum(Om, Bp, value),
                _resolvent_at_zeta(pair_block, W0bar, complex(value) ** 2),
            )
            for value in z_values
        ]
        # This is deliberately another direct WP1+solve evaluation, not a
        # contour reconstruction or an alias of a sample slot.
        exact_static = _resolvent_at_zeta(pair_block, W0bar, 0.0j)
        static_error = _relative_error(
            evaluate_pole_sum(Om, Bp, 0.0j), exact_static)
        error = max(errors, default=0.0)
        selected = (Om, Bp, error, static_error,
                    fold_el, drop_el, width, closure)
        if error <= float(sample_rel_tol) and static_error <= STATIC_REL_TOL:
            break
        # Empty contours are dropped by the compression, so the stored pole
        # count can lag the interval count; bound BOTH or the bisection can
        # run past the store's allocated maximum without ever tripping.
        if (int(Om.shape[0]) >= MAX_CLUSTERS
                or len(intervals) >= MAX_CLUSTERS):
            raise ValueError(
                "GATE intraband_cluster_budget: six contour clusters do not "
                "meet the fixed block certificate; "
                f"intervals={len(intervals)}, stored_poles={int(Om.shape[0])}, "
                f"sample_max_rel={error:.6e}, static_max_rel="
                f"{static_error:.6e}, allowed_sample={float(sample_rel_tol):.6e}, "
                f"allowed_static={STATIC_REL_TOL:.6e}")
        intervals = _split_largest_trace_interval(intervals, M, pair_block)

    (Om, Bp, error, static_error,
     fold_el, drop_el, width, closure) = selected
    n_poles = int(Om.shape[0])
    _print_memory_model(pair_block, W0bar, n_poles)
    return IntrabandRow(
        Omega_p=Om,
        B_p=Bp,
        n_poles=n_poles,
        n_modes=n_pair,
        sample_max_rel_error=float(error),
        static_max_rel_error=float(static_error),
        certified=True,
        folded_modes=0,
        dropped_modes=0,
        folded_elements=int(fold_el),
        dropped_elements=int(drop_el),
        cluster_width_max_ry=float(width),
        zero_mode_weight=float(closure.zero_mode_weight),
        zero_mode_cluster=int(closure.zero_mode_cluster),
        zero_mode_pole_shift=float(closure.zero_mode_pole_shift),
    )


def pad_row(row, n_poles):
    """Pad a shorter certified q row with causal, exactly dark poles."""
    target = int(n_poles)
    if target < row.n_poles:
        raise ValueError(f"cannot pad {row.n_poles} intraband poles to {target}")
    if target == row.n_poles:
        return row.Omega_p, row.B_p
    n_mu = int(row.Omega_p.shape[-1])
    mesh = _mesh_of(row.Omega_p, "pad_row")
    sharding = NamedSharding(mesh, P(None, "x", "y"))
    extra = target - row.n_poles
    sentinel = jnp.ones((extra, n_mu, n_mu), dtype=jnp.complex128)
    dark = jnp.zeros((extra, n_mu, n_mu), dtype=jnp.complex128)
    return (
        jax.lax.with_sharding_constraint(
            jnp.concatenate((row.Omega_p, sentinel), axis=0), sharding),
        jax.lax.with_sharding_constraint(
            jnp.concatenate((row.B_p, dark), axis=0), sharding),
    )


__all__ = [
    "ClusterClosure",
    "IntrabandRow",
    "MAX_CLUSTERS",
    "MODEL",
    "MOMENT_REL_TOL",
    "SAMPLE_REL_TOL",
    "STATIC_REL_TOL",
    "SUM_RULE_REL_TOL",
    "build_row",
    "evaluate_pole_sum",
    "pad_row",
]

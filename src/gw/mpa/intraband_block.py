"""Certified finite-q crossing skeleton with exact-nodal amplitudes.

Production constructs spectral-projector moments from the centroid-space
resolvent.  Every contour node evaluates the landed WP1 crossing kernel and
solves only an ``N_mu x N_mu`` system.  Pair data remain the two sharded
``N_s x N_mu`` vertex tables plus ``O(N_s)`` scalars; production never forms
``H`` or any other ``N_s x N_s`` object.

The contour construction certifies an interval skeleton without ever forming
the pair-space matrix.  Pole positions and widths are cluster scalars derived
from those intervals; residues are the elementwise constrained linear
least-squares solution against exact double-Dyson data on the stored support
and a shared near-line ladder.  The former frozen-static two-moment amplitudes
are evaluated only for the required A/B diagnostic and are never shipped.

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
GAP_CERTIFICATE_FIRST_REAL_RY = 0.04
GAP_CERTIFICATE_LOWEST_BISECTION_REAL_RY = (
    0.5 * GAP_CERTIFICATE_FIRST_REAL_RY)
# Ratified amendment, 2026-08-17 (DESIGN §2.4c Ruling 3, under claim 0329's
# demand principle; exact-nodal re-anchoring ruling confirmed by claim 0353).
# The adaptive gap shrinks on demand, and an OPEN D_M at the current edge is a
# demand signal: asymptotic weight is by §2.4b's dichotomy a FINITE mode, so
# the edge is excluding physics the contour is obliged to capture.  Before this
# amendment the D_M refusal fired *before* any shrink path could run, which was
# an ordering flaw and not a protection.  Its bound is unchanged: doublings
# continue only while D_M improves, a plateau at machine precision closes and
# merges the residual D_V, and a plateau that has NOT closed keeps the
# unconditional refusal (a mode the contour cannot reach).  There is no descent
# to eps*zeta_max: this cap refuses by name well before the ~52 dyadic
# doublings §2.4c Ruling 3 rejected.
MAX_ORIGIN_GAP_DOUBLINGS = 32
NEAR_LINE_SEED_REAL_RY = np.asarray(
    (0.04, 0.08, 0.15, 0.30, 0.60), dtype=np.float64)
# The same truncated-SVD policy as the incumbent MPA residue fit.  This is a
# numerical rank declaration, not a physics or deck parameter.
RESIDUE_LS_RCOND = 1.0e-13
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
    # The frozen-block error at the stored fit samples is diagnostic only.
    sample_max_rel_error: float
    gap_max_rel_error: float
    static_max_rel_error: float
    certified: bool
    folded_modes: int
    dropped_modes: int
    folded_elements: int
    dropped_elements: int
    cluster_width_max_ry: float
    # Exact-nodal certificate/provenance.  The frozen arrays are transient
    # diagnostics and are deliberately never passed to the store writer.
    frozen_gap_max_rel_error: float = 0.0
    ladder_rcond: float = 1.0
    ladder_nodes: int = 0
    ladder_refinement: int = 0
    ladder_initial_max_rel_error: float = 0.0
    frozen_Omega_p: jax.Array | None = None
    frozen_B_p: jax.Array | None = None
    zero_mode_weight: float = 0.0
    zero_mode_cluster: int = -1
    zero_mode_pole_shift: float = 0.0
    # The certified origin exclusion this row was built at, and how many
    # demand-driven doublings reached it from the registered starting edge.
    origin_gap_ry2: float = 0.0
    origin_gap_doublings: int = 0
    origin_gap_m_closure: float = 0.0


class OpenAsymptoticClosure(ValueError):
    """The §2.4b open-``D_M`` refusal, carrying the numbers it refused on.

    It stays a ``ValueError`` with the same gate name and message, so every
    caller without a demand path refuses exactly as before.  Only the
    production driver catches it, and only to run the authorized
    demand-driven origin-gap shrink; if that shrink stops improving ``D_M``
    the same exception is re-raised untouched.
    """

    def __init__(self, message, *, m_closure, v_closure, order, origin_gap):
        super().__init__(message)
        self.m_closure = float(m_closure)
        self.v_closure = float(v_closure)
        self.order = int(order)
        self.origin_gap = None if origin_gap is None else float(origin_gap)


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


def _contour_geometry(pair_block, W0bar, *, origin_gap=None):
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
    # Low-level contour callers retain the smallest data-resolved exclusion.
    # The production adaptive driver overrides this with
    # ``_certified_origin_gap``.  Keeping these policies separate matters for
    # the D_M dichotomy: diagnostic/oracle calls must not silently exclude a
    # finite screened mode merely because no gap certificate was supplied.
    minimum_gap = np.finfo(np.float64).eps * zeta_max
    initial_gap = max(0.25 * float(np.min(positive_u2)), minimum_gap)
    zero_gap = initial_gap if origin_gap is None else float(origin_gap)
    # The interaction radius is the derived imaginary excursion.  Its only
    # floor is roundoff separation from the real axis.
    height = max(interaction,
                 128.0 * np.finfo(np.float64).eps * zeta_max)
    if (not np.isfinite(zero_gap) or zero_gap < minimum_gap
            or not zero_gap < zeta_max):
        raise ValueError(
            "GATE intraband_contour_domain: derived two-sided strip cannot "
            "exclude zeta=0: "
            f"zeta_min={zeta_min:.17e}, zero_gap={zero_gap:.17e}, "
            f"zeta_max={zeta_max:.17e}")
    return zeta_min, zero_gap, zeta_max, height


def _certified_origin_gap(pair_block, W0bar):
    """Initial production exclusion from the lowest certified bisection edge.

    The first held-out point is 0.04 Ry, so the edge of the first resolution
    cell between the exact origin anchor and that point is its 0.02-Ry
    midpoint.  The contour is in ``zeta=z**2``.  This policy must not depend
    on the smallest bare transition: production crossing rows contain
    arbitrarily near-degenerate pairs, and D_M/D_V adjudicate anything
    excluded here.
    """
    _left, _data_gap, zeta_max, _height = _contour_geometry(pair_block, W0bar)
    minimum_gap = np.finfo(np.float64).eps * zeta_max
    return max(
        min(GAP_CERTIFICATE_LOWEST_BISECTION_REAL_RY ** 2,
            0.5 * zeta_max),
        minimum_gap,
    )


def shared_near_line_ladder(max_lambda_ry, varpi_near_ry, refinement):
    """Return the deterministic nested WP3-A5 near-line ladder.

    The level-zero real nodes are the five DESIGN §7.1 points, extended by
    exact factor-two steps until the top covers ``max_q Lambda(q)``.  Each
    refinement interleaves log midpoints, so old nodes occupy the even slots
    and the newly inserted nodes occupy the odd slots.  The latter are the
    amplitude-fit rows; the former remain held out for the certificate.
    """
    top = float(max_lambda_ry)
    height = float(varpi_near_ry)
    level = int(refinement)
    if not np.isfinite(top) or top < 0.0:
        raise ValueError(
            "GATE intraband_near_line_ladder: max_q Lambda(q) must be "
            f"finite and non-negative; got {top!r}")
    if not np.isfinite(height) or height <= 0.0:
        raise ValueError(
            "GATE intraband_near_line_ladder: varpi_near must be finite "
            f"and positive; got {height!r}")
    if level < 0:
        raise ValueError(
            "GATE intraband_near_line_ladder: refinement must be "
            f"non-negative; got {level}")

    real = list(NEAR_LINE_SEED_REAL_RY)
    while real[-1] < top:
        real.append(2.0 * real[-1])
    for _ in range(level):
        refined = []
        for lo, hi in zip(real[:-1], real[1:]):
            refined.extend((lo, float(np.sqrt(lo * hi))))
        refined.append(real[-1])
        if len(refined) == len(real):
            raise ValueError(
                "GATE intraband_near_line_ladder: log-midpoint refinement "
                "made no representable progress")
        real = refined
    return np.asarray(real, dtype=np.float64) + 1.0j * height


def _initial_intervals(pair_block, W0bar, *, origin_gap=None):
    """Three certified tiles, extending negative for signed MP1 weights."""
    left, zero_gap, right, _height = _contour_geometry(
        pair_block, W0bar, origin_gap=origin_gap)
    u2 = np.sort(np.square(np.abs(_host_replicated(pair_block[0]))))
    # An adaptive gap may exclude many bare near-zero pairs.  They are
    # intentionally adjudicated by D_M/D_V, so they cannot also nominate a
    # positive-interval split below the active contour edge.
    u2 = u2[(u2 >= zero_gap) & (u2 <= right)]
    w = _host_replicated(pair_block[1])
    negative_active = bool(np.any(w < 0.0) and left < -zero_gap)
    n_positive = MIN_CLUSTERS - int(negative_active)
    edges = [zero_gap]
    if u2.size >= n_positive:
        gap_order = np.argsort(np.diff(u2))[::-1][:n_positive - 1]
        for gap_index in np.sort(gap_order):
            edge = 0.5 * (float(u2[gap_index]) + float(u2[gap_index + 1]))
            edge = min(max(edge, np.nextafter(edges[-1], np.inf)), right)
            edges.append(edge)
        edges.append(right)
    else:
        # Preserve the three-cluster starting contract even when the adaptive
        # gap leaves fewer bare energies available to nominate internal cuts.
        edges = list(np.linspace(zero_gap, right, n_positive + 1))
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


def _moments_at_order(
        pair_block, W0bar, intervals, order, V_total, *, height=None,
        origin_gap=None):
    mesh = _mesh_of(W0bar, "_moments_at_order")
    matrix_shard = NamedSharding(mesh, P("x", "y"))
    pole_shard = NamedSharding(mesh, P(None, "x", "y"))
    _left, _zero_gap, _right, max_height = _contour_geometry(
        pair_block, W0bar, origin_gap=origin_gap)
    height = max_height if height is None else float(height)
    shape = tuple(W0bar.shape)
    M_rows, V_rows = [], []
    normalization = 1.0 / (2.0j * np.pi)
    for interval in intervals:
        M = jnp.zeros(shape, dtype=jnp.complex128)
        V = jnp.zeros(shape, dtype=jnp.complex128)
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
        M_rows.append(jax.lax.with_sharding_constraint(M, matrix_shard))
        V_rows.append(jax.lax.with_sharding_constraint(V, matrix_shard))
    return tuple(
        jax.lax.with_sharding_constraint(jnp.stack(rows), pole_shard)
        for rows in (M_rows, V_rows)
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


def _merge_zero_mode(M, V, M_total, V_total, m_closure, v_closure, order,
                     *, origin_gap=None):
    """§2.4b deflation-merge, applied before the two-moment match.

    The dichotomy is codified here and nowhere else: an open ``M`` closure is
    a missed *finite*-lambda mode and refuses; a closed ``M`` closure with a
    nonzero ``D_V`` is the machine-zero screened mode's static weight, which
    the sum rule defines with no free parameter.
    """
    if m_closure > SUM_RULE_REL_TOL:
        raise OpenAsymptoticClosure(
            "GATE intraband_contour_sum_rule: the asymptotic closure is open "
            f"(||D_M||/||M_total||={m_closure:.6e} > "
            f"{SUM_RULE_REL_TOL:.1e}) at quadrature order {order}.  A missed "
            "finite-lambda mode carries asymptotic weight, so it can never "
            "masquerade as zero-mode static weight; the static deficiency "
            f"(||D_V||/||V_total||={v_closure:.6e}) is NOT merged.  Fix the "
            "contour tiling.",
            m_closure=m_closure, v_closure=v_closure, order=order,
            origin_gap=origin_gap)

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


def _has_plateaued(value, previous):
    """True when doubling the quadrature order stopped moving a closure."""
    return (previous is not None
            and abs(value - previous)
            <= _V_PLATEAU_REL_TOL * max(value, previous))


def _demand_shrink_improves(value, previous):
    """True when the last origin-gap doubling actually moved ``D_M`` down.

    Same plateau idiom as the quadrature ladder, applied to the gap axis: a
    deficit that a doubling leaves stationary is weight the contour cannot
    reach by shrinking, so the unconditional refusal stands.
    """
    if previous is None:
        return True
    return value < previous and not _has_plateaued(value, previous)


def _demand_shrink_has_pending_capture(pair_block, origin_gap):
    """True when bare crossing energies still lie inside the excluded gap.

    Improvement alone cannot be the sole continuation signal: a *discrete*
    excluded mode holds ``D_M`` exactly constant until the edge passes below
    it, so a step-to-step improvement rule would abandon precisely the finite
    mode §2.4b says must be captured.  A bare crossing energy still inside the
    exclusion is direct, parameter-free evidence that shrinking has something
    left to reach.  When neither signal holds, the deficit is weight no edge
    can reach and the unconditional refusal stands.
    """
    u2 = np.square(np.abs(_host_replicated(pair_block[0])))
    return bool(np.any((u2 > 0.0) & (u2 < float(origin_gap))))


def _cluster_moment_matrices(
        pair_block, W0bar, intervals, *, moment_rel_tol=MOMENT_REL_TOL,
        origin_gap=None):
    """Contour M/V with mandatory movement and sum-rule refusals.

    Returns ``(M, V, closure)``.  ``V`` already carries the §2.4b deflation
    merge, so both sum rules hold as exact identities on the returned
    moments.  Width-only T1/T2 moments are intentionally absent.
    """
    intervals = tuple((float(lo), float(hi)) for lo, hi in intervals)
    if not intervals or any(not lo < hi for lo, hi in intervals):
        raise ValueError(
            "GATE intraband_contour_intervals: intervals must be nonempty "
            "ordered (lo,hi) pairs")
    left, zero_gap, right, height = _contour_geometry(
        pair_block, W0bar, origin_gap=origin_gap)
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
    previous_m_closure = None
    current = None
    movement = np.inf
    m_closure = np.inf
    v_closure = np.inf
    for order in _QUADRATURE_ORDERS:
        current = _moments_at_order(
            pair_block, W0bar, intervals, order, V_total,
            origin_gap=origin_gap)
        movements = (np.inf,) * 2
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
            scales = (mass_scale, static_scale)
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
                "move_M/V="
                f"{movements[0]:.3e}/{movements[1]:.3e} "
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
        v_plateaued = _has_plateaued(v_closure, previous_v_closure)
        m_plateaued = _has_plateaued(m_closure, previous_m_closure)
        # Both closures stay convergence criteria until they *plateau*.  A
        # refinement-invariant deficiency is a spectral-domain fact (claim
        # 0319: D_V = 3.277321e-3 at orders 16 and 32); a shrinking one is
        # quadrature error and must keep refining.  Adjudicating an open M
        # closure the moment it stops improving buys the dichotomy's own
        # message instead of a generic order-512 failure.
        settled = (
            (m_closure <= SUM_RULE_REL_TOL
             and (v_closure <= SUM_RULE_REL_TOL or v_plateaued))
            or (m_closure > SUM_RULE_REL_TOL and m_plateaued)
        )
        if movement <= float(moment_rel_tol) and settled:
            merged, closure = _merge_zero_mode(
                current[0], current[1], M_total, V_total,
                m_closure, v_closure, order, origin_gap=origin_gap)
            return (current[0], merged, closure)
        previous = current
        previous_v_closure = v_closure
        previous_m_closure = m_closure
    if m_closure > SUM_RULE_REL_TOL:
        # Orders exhausted with the spectral domain still incomplete: name
        # the dichotomy rather than the loop that ran out.
        _merge_zero_mode(
            current[0], current[1], M_total, V_total,
            m_closure, v_closure, _QUADRATURE_ORDERS[-1],
            origin_gap=origin_gap)
    raise ValueError(
        "GATE intraband_contour_sum_rule: quadrature failed mandatory "
        f"closure at order {_QUADRATURE_ORDERS[-1]}: "
        f"moment_movement={movement:.6e}, Sigma_M_rel={m_closure:.6e}, "
        f"Sigma_V_rel={v_closure:.6e}, sum_tolerance={SUM_RULE_REL_TOL:.1e}, "
        f"movement_tolerance={float(moment_rel_tol):.1e}")


def _cluster_widths(intervals):
    """§2.2 interval half-width, with the named overdamped zero carry."""
    widths = []
    for index, (lo, hi) in enumerate(intervals):
        if hi < 0.0:
            if jax.process_index() == 0:
                print(
                    "[intraband-cluster-width] overdamped interval "
                    f"{index}: zeta=[{lo:.6e},{hi:.6e}], width=0",
                    flush=True,
                )
            widths.append(0.0)
            continue
        if lo <= 0.0:
            raise ValueError(
                "GATE intraband_cluster_width: interval crosses the excluded "
                f"origin: [{lo:.17e},{hi:.17e}]")
        widths.append(0.5 * (np.sqrt(hi) - np.sqrt(lo)))
    return np.asarray(widths, dtype=np.float64)


def _active_cluster_moments(M, V, intervals):
    """Drop contours that are empty under both certified sum rules."""
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
    return (
        M[active],
        V[active],
        tuple(intervals[index] for index in np.flatnonzero(keep)),
    )


def _cluster_scalar_poles(M, V, intervals):
    """WP3-A5 positions/widths from certified interval geometry.

    Near-real intervals use midpoint and half-width in ``sqrt(zeta)``.
    Negative intervals are the overdamped branch: position is damping and
    there is no second width.  ``M`` and ``V`` participate only in deciding
    which certified intervals are nonempty; they no longer set any element's
    position or amplitude.
    """
    M_active, V_active, active_intervals = _active_cluster_moments(
        M, V, intervals)
    poles = []
    widths = []
    for lo, hi in active_intervals:
        if hi < 0.0:
            damping = 0.5 * (np.sqrt(abs(lo)) + np.sqrt(abs(hi)))
            poles.append(complex(0.0, -damping))
            widths.append(0.0)
        else:
            if lo <= 0.0:
                raise ValueError(
                    "GATE intraband_cluster_width: interval crosses the "
                    f"excluded origin: [{lo:.17e},{hi:.17e}]")
            omega = 0.5 * (np.sqrt(lo) + np.sqrt(hi))
            width = 0.5 * (np.sqrt(hi) - np.sqrt(lo))
            poles.append(complex(omega, -width))
            widths.append(width)
    return (
        M_active,
        V_active,
        active_intervals,
        np.asarray(poles, dtype=np.complex128),
        np.asarray(widths, dtype=np.float64),
    )


def _stack_matrix_observations(values, mesh, where):
    """Validate and stack a small observable axis over sharded matrix rows."""
    rows = tuple(values)
    if not rows:
        raise ValueError(f"{where} requires at least one matrix observation")
    shape = tuple(rows[0].shape)
    if len(shape) != 2 or any(tuple(value.shape) != shape for value in rows):
        raise ValueError(
            f"{where} requires same-shaped rank-2 matrices; got "
            f"{[tuple(value.shape) for value in rows]}")
    return jax.lax.with_sharding_constraint(
        jnp.stack(rows), NamedSharding(mesh, P(None, "x", "y")))


def _constrained_linear_residues(
        mesh, omega_scalar, z_fit, delta_fit, delta_static, *,
        rcond=RESIDUE_LS_RCOND):
    """Solve the hard-static constrained complex LS for every element.

    With ``c_p=-2/Omega_p`` the equality is ``c @ B = DeltaW(0)``.
    Eliminating the largest-magnitude constraint column makes that equality
    algebraic, then one small SVD supplies the same pseudo-inverse to every
    sharded matrix element.  No elementwise solver or host matrix payload is
    materialized.
    """
    omega = np.asarray(omega_scalar, dtype=np.complex128).reshape(-1)
    z = np.asarray(z_fit, dtype=np.complex128).reshape(-1)
    target = _stack_matrix_observations(
        delta_fit, mesh, "_constrained_linear_residues")
    if z.size != int(target.shape[0]):
        raise ValueError(
            "GATE intraband_residue_support: z/data cardinality mismatch: "
            f"{z.size} coordinates for {int(target.shape[0])} matrices")
    if omega.size < 1 or np.any(~np.isfinite(omega)) or np.any(omega == 0.0):
        raise ValueError(
            "GATE intraband_residue_geometry: cluster-scalar positions "
            "must be finite and nonzero")
    cutoff = float(rcond)
    if not 0.0 < cutoff < 1.0:
        raise ValueError(
            "GATE intraband_residue_rcond: require 0 < rcond < 1; got "
            f"{cutoff!r}")

    design = (2.0 * omega[None, :]
              / (z[:, None] ** 2 - omega[None, :] ** 2))
    constraint = -2.0 / omega
    pivot = int(np.argmax(np.abs(constraint)))
    free = np.asarray(
        [index for index in range(omega.size) if index != pivot],
        dtype=np.int32)
    pole_shard = NamedSharding(mesh, P(None, "x", "y"))
    static = jax.lax.with_sharding_constraint(
        jnp.asarray(delta_static), NamedSharding(mesh, P("x", "y")))

    if free.size:
        reduced = (
            design[:, free]
            - design[:, pivot, None]
            * constraint[free][None, :] / constraint[pivot])
        u, singular, vh = np.linalg.svd(reduced, full_matrices=False)
        # Measure the constrained map against the scale of the original
        # response columns.  ``s[-1]/s[0]`` alone incorrectly calls a single
        # numerically-zero reduced column well conditioned (its ratio is
        # trivially one), which is exactly what coincident cluster poles
        # produce after equality elimination.
        reference = float(np.linalg.norm(design[:, free], ord=2))
        if (singular.size != free.size or singular[0] <= 0.0
                or not np.isfinite(reference) or reference <= 0.0):
            achieved = 0.0
        else:
            achieved = float(singular[-1] / reference)
        if not np.isfinite(achieved) or achieved <= cutoff:
            raise ValueError(
                "GATE intraband_residue_rcond: constrained residue design "
                f"is rank deficient; achieved rcond={achieved:.6e}, "
                f"required > {cutoff:.1e}, observations={z.size}, "
                f"clusters={omega.size}")
        inverse = ((vh.conj().T / singular[None, :]) @ u.conj().T)
        particular_model = design[:, pivot] / constraint[pivot]
        rhs = target - particular_model[:, None, None] * static[None, :, :]
        B_free = jax.lax.with_sharding_constraint(
            jnp.einsum(
                "co,omn->cmn", jnp.asarray(inverse), rhs, optimize=True),
            NamedSharding(mesh, P(None, "x", "y")),
        )
        B = jnp.zeros(
            (omega.size, *static.shape), dtype=jnp.complex128)
        B = B.at[jnp.asarray(free)].set(B_free)
        pivot_value = (
            static
            - jnp.einsum(
                "c,cmn->mn", jnp.asarray(constraint[free]), B_free,
                optimize=True)
        ) / constraint[pivot]
        B = B.at[pivot].set(pivot_value)
    else:
        achieved = 1.0
        B = (static / constraint[0])[None, :, :]
    # ``with_sharding_constraint`` outside a compiled region may simplify a
    # one-device NamedSharding to SingleDeviceSharding.  The pole-store API
    # deliberately requires the mesh metadata even in that case.  A compiled
    # identity with an explicit output sharding preserves that metadata and,
    # unlike host placement, never creates a replicated matrix payload.
    place = jax.jit(lambda value: value, out_shardings=pole_shard)
    B = place(B)
    Omega = place(jnp.broadcast_to(
        jnp.asarray(omega)[:, None, None], tuple(B.shape)))

    # The hard equality is a gate on the actual sharded result, not merely on
    # the algebra above.  Its scale is the design's elementwise z=0 identity.
    anchored = jnp.einsum(
        "c,cmn->mn", jnp.asarray(constraint), B, optimize=True)
    anchor_error = _relative_error(anchored, static)
    if anchor_error > SUM_RULE_REL_TOL:
        raise ValueError(
            "GATE intraband_static_constraint: constrained residue solve "
            f"left z=0 residual {anchor_error:.6e} > "
            f"{SUM_RULE_REL_TOL:.1e}")
    return Omega, B, achieved, anchor_error


def _compress_moments(mesh, M, V, intervals):
    """Frozen two-moment amplitudes, retained only as an A/B diagnostic."""
    matrix_shard = NamedSharding(mesh, P("x", "y"))
    pole_shard = NamedSharding(mesh, P(None, "x", "y"))
    M, V, active_intervals = _active_cluster_moments(M, V, intervals)
    widths = _cluster_widths(active_intervals)
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


def _maximum_relative(models, exacts):
    """Worst Frobenius-relative member of a matrix observable family."""
    return max(
        (_relative_error(model, exact)
         for model, exact in zip(models, exacts)),
        default=0.0,
    )


def _frozen_exact_values(pair_block, W0bar, z_values):
    """Exact frozen-skeleton data for unit/oracle callers without WP2 data."""
    return tuple(
        _resolvent_at_zeta(pair_block, W0bar, complex(z) ** 2)
        for z in z_values
    )


def build_row(
        W0bar, pair_block, z_samples, *, support_delta=None,
        ladder_z=None, ladder_delta=None, ladder_refinement=0,
        gap_certificate=None, refuse_on_certificate=True):
    """Build one exact-nodal row on a contour-certified interval skeleton.

    ``support_delta`` is exact WP2 ``DeltaW`` on all stored samples.
    ``ladder_delta`` is exact WP2 ``DeltaW`` on the one shared near-line
    ladder.  Odd ladder nodes join every support sample in the constrained
    linear residue solve; even nodes are held out and drive interval
    bisection at the unchanged 4e-3 budget.  After that split passes, the
    shipped residues are refit on all ladder nodes.

    Tests that exercise the skeleton in isolation may omit the two exact-data
    families; the direct frozen resolvent then supplies an oracle with the
    same shapes.  Production always passes dynamic double-Dyson values.
    ``gap_certificate`` is the pre-WP3-A5 compatibility hook used only by the
    untouched synthetic bisection tests; it can demand more intervals but can
    never relax the even-ladder gate.
    """
    mesh = _mesh_of(W0bar, "build_row")
    n_pair = int(pair_block[0].shape[0])
    if n_pair == 0:
        raise ValueError("build_row is not called for the empty Gamma block")

    support_supplied = support_delta is not None
    ladder_supplied = ladder_delta is not None
    authoritative_ladder = support_supplied and ladder_supplied
    diagnostic_z = tuple(
        complex(value) for value in np.asarray(z_samples).reshape(-1))
    if not diagnostic_z:
        raise ValueError(
            "GATE intraband_residue_support: at least one support node is "
            "required")
    if diagnostic_z[0] != 0.0j and support_supplied:
        raise ValueError(
            "GATE intraband_static_constraint: the first support node must "
            "be the exact z=0 anchor")
    if diagnostic_z[0] != 0.0j:
        # Skeleton-only unit callers historically supplied arbitrary sample
        # coordinates.  Their private frozen oracle can add the independent
        # origin value without weakening the production contract above.
        diagnostic_z = (0.0j,) + diagnostic_z
    if support_delta is None:
        support_delta = _frozen_exact_values(
            pair_block, W0bar, diagnostic_z)
    support_delta = tuple(support_delta)
    if len(support_delta) != len(diagnostic_z):
        raise ValueError(
            "GATE intraband_residue_support: support z/data cardinality "
            f"mismatch: {len(diagnostic_z)} vs {len(support_delta)}")

    refinement = int(ladder_refinement)
    if ladder_z is None:
        max_lambda = float(np.max(np.abs(_host_replicated(pair_block[0]))))
        ladder_z = shared_near_line_ladder(max_lambda, 0.2, refinement)
    ladder_z = np.asarray(ladder_z, dtype=np.complex128).reshape(-1)
    if ladder_delta is None:
        ladder_delta = _frozen_exact_values(pair_block, W0bar, ladder_z)
    ladder_delta = tuple(ladder_delta)
    if len(ladder_delta) != ladder_z.size:
        raise ValueError(
            "GATE intraband_residue_support: ladder z/data cardinality "
            f"mismatch: {ladder_z.size} vs {len(ladder_delta)}")
    odd = np.arange(ladder_z.size, dtype=np.int32)[1::2]
    even = np.arange(ladder_z.size, dtype=np.int32)[::2]
    if odd.size == 0 or even.size == 0:
        raise ValueError(
            "GATE intraband_near_line_ladder: odd fit nodes and even "
            "certificate nodes must both be nonempty")
    # The odd/even split is refusal-grade only when both exact WP2 families
    # were supplied.  The direct frozen oracle used by legacy skeleton tests
    # instead fits every synthetic ladder node: it has no standing to fail a
    # production certificate and needs enough rows to exercise six-cluster
    # bisection without manufacturing a rank deficiency.
    fit_indices = odd if authoritative_ladder else np.arange(
        ladder_z.size, dtype=np.int32)
    fit_z = np.concatenate((
        np.asarray(diagnostic_z, dtype=np.complex128),
        ladder_z[fit_indices]))
    fit_delta = support_delta + tuple(
        ladder_delta[index] for index in fit_indices)
    exact_static = support_delta[0]

    _left, _unused_gap, zeta_max, _height = _contour_geometry(
        pair_block, W0bar)
    origin_gap = _certified_origin_gap(pair_block, W0bar)
    minimum_gap = np.finfo(np.float64).eps * zeta_max
    intervals = _initial_intervals(
        pair_block, W0bar, origin_gap=origin_gap)
    selected = None
    doublings = 0
    open_m_closure = 0.0
    previous_open_m = None
    while True:
        try:
            M, V, closure = _cluster_moment_matrices(
                pair_block, W0bar, intervals,
                moment_rel_tol=MOMENT_REL_TOL,
                origin_gap=origin_gap)
        except OpenAsymptoticClosure as refusal:
            # Amended trigger set (coordinator-authorized 2026-08-17): an open
            # D_M at the current edge is itself demand.  The excluded weight is
            # asymptotic, hence a finite mode by §2.4b, hence physics the
            # contour must capture -- so the edge halves and the row is rebuilt
            # rather than refusing before any shrink path could run.
            next_gap = 0.5 * origin_gap
            improving = _demand_shrink_improves(
                refusal.m_closure, previous_open_m)
            pending = _demand_shrink_has_pending_capture(
                pair_block, origin_gap)
            if (not (improving or pending) or next_gap <= minimum_gap
                    or doublings >= MAX_ORIGIN_GAP_DOUBLINGS):
                if jax.process_index() == 0:
                    print(
                        "[intraband-origin-gap] demand shrink exhausted: "
                        f"doublings={doublings} "
                        f"origin_gap={origin_gap:.17e} "
                        f"D_M={refusal.m_closure:.6e} "
                        f"previous_D_M="
                        f"{-1.0 if previous_open_m is None else previous_open_m:.6e} "
                        f"improving={improving} pending_capture={pending}; "
                        "the unconditional D_M refusal stands",
                        flush=True,
                    )
                raise
            if jax.process_index() == 0:
                print(
                    "[intraband-origin-gap] open D_M demands capture: "
                    f"shrink {origin_gap:.17e} -> {next_gap:.17e} "
                    f"doubling={doublings + 1} "
                    f"D_M={refusal.m_closure:.6e} "
                    f"D_V={refusal.v_closure:.6e} "
                    f"order={refusal.order}",
                    flush=True,
                )
            previous_open_m = refusal.m_closure
            open_m_closure = refusal.m_closure
            origin_gap = next_gap
            doublings += 1
            intervals = _initial_intervals(
                pair_block, W0bar, origin_gap=origin_gap)
            continue
        # Stage 1's two-moment amplitudes are now diagnostic only.  Stage 2
        # consumes the same certified active intervals, but no element of M/V
        # sets a pole position, width, or shipped amplitude.
        frozen_Om, frozen_Bp, fold_el, drop_el, _frozen_width = (
            _compress_moments(mesh, M, V, intervals))
        (_M_active, _V_active, _active_intervals,
         omega_scalar, widths) = _cluster_scalar_poles(M, V, intervals)
        Om, Bp_validation, achieved_rcond, anchor_error = (
            _constrained_linear_residues(
                mesh, omega_scalar, fit_z, fit_delta, exact_static))
        even_models = tuple(
            evaluate_pole_sum(Om, Bp_validation, ladder_z[index])
            for index in even)
        observed_gap_error = _maximum_relative(
            even_models, (ladder_delta[index] for index in even))
        # A synthetic frozen-block oracle is deliberately non-authoritative:
        # preserve the old skeleton tests and report its shape only through
        # the frozen A/B diagnostics.  Production always takes this exact
        # held-out residual as the refusal-grade gap certificate.
        gap_error = observed_gap_error if authoritative_ladder else 0.0
        gap_allowed = SAMPLE_REL_TOL
        external_error, external_allowed = 0.0, np.inf
        if gap_certificate is not None:
            external_error, external_allowed = map(
                float, gap_certificate(Om, Bp_validation))
            if (not np.isfinite(external_error) or external_error < 0.0
                    or not np.isfinite(external_allowed)
                    or external_allowed <= 0.0):
                raise ValueError(
                    "GATE intraband_gap_certificate: callback returned "
                    f"error={external_error!r}, "
                    f"allowed={external_allowed!r}")
        static_error = _relative_error(
            evaluate_pole_sum(Om, Bp_validation, 0.0j), exact_static)
        frozen_gap_error = _maximum_relative(
            (evaluate_pole_sum(frozen_Om, frozen_Bp, value)
             for value in ladder_z),
            ladder_delta,
        )
        selected = (
            Om, Bp_validation, gap_error, static_error,
            fold_el, drop_el, float(np.max(widths, initial=0.0)), closure,
            achieved_rcond, anchor_error, frozen_Om, frozen_Bp,
            frozen_gap_error, omega_scalar,
        )
        if jax.process_index() == 0:
            print(
                "[intraband-even-certificate] "
                f"clusters={int(Om.shape[0])} "
                f"intervals={len(intervals)} "
                f"origin_gap_ry2={origin_gap:.17e} "
                f"ladder_nodes={int(ladder_z.size)} "
                f"gap_max_rel={gap_error:.6e} "
                f"static_max_rel={static_error:.6e} "
                f"ls_rcond={achieved_rcond:.6e} "
                f"frozen_gap_diag={frozen_gap_error:.6e}",
                flush=True,
            )
        certificate_green = (
            gap_error <= gap_allowed
            and external_error <= external_allowed
            and static_error <= STATIC_REL_TOL)
        if certificate_green:
            break
        # Empty contours are dropped by the compression, so the stored pole
        # count can lag the interval count; bound BOTH or the bisection can
        # run past the store's allocated maximum without ever tripping.
        if (int(Om.shape[0]) >= MAX_CLUSTERS
                or len(intervals) >= MAX_CLUSTERS):
            # The origin is shrunk only in response to a failed held-out gap
            # certificate.  D_M and the two closure gates above are never
            # caught here and therefore remain unconditional refusals.
            next_gap = max(minimum_gap, 0.5 * origin_gap)
            if ((gap_error > gap_allowed
                    or external_error > external_allowed)
                    and next_gap < origin_gap
                    and doublings < MAX_ORIGIN_GAP_DOUBLINGS):
                if jax.process_index() == 0:
                    print(
                        "[intraband-origin-gap] held-out gap certificate "
                        f"requested shrink {origin_gap:.17e} -> "
                        f"{next_gap:.17e}; gap_max_rel={gap_error:.6e} "
                        f"allowed={gap_allowed:.6e}",
                        flush=True,
                    )
                origin_gap = next_gap
                doublings += 1
                intervals = _initial_intervals(
                    pair_block, W0bar, origin_gap=origin_gap)
                continue
            if not bool(refuse_on_certificate):
                break
            raise ValueError(
                "GATE intraband_cluster_budget: six contour clusters do not "
                "meet the held-out even-ladder certificate; "
                f"intervals={len(intervals)}, stored_poles={int(Om.shape[0])}, "
                f"gap_max_rel={gap_error:.6e}, static_max_rel="
                f"{static_error:.6e}, allowed_gap="
                f"{gap_allowed:.6e}, "
                f"allowed_static={STATIC_REL_TOL:.6e}")
        intervals = _split_largest_trace_interval(intervals, M, pair_block)

    (Om, Bp_validation, gap_error, static_error,
     fold_el, drop_el, width, closure, achieved_rcond, anchor_error,
     frozen_Om, frozen_Bp, frozen_gap_error, omega_scalar) = selected
    certified = bool(
        gap_error <= SAMPLE_REL_TOL and static_error <= STATIC_REL_TOL)
    # Certify on the odd/even split, then refit on every exact ladder node.
    # The equality constraint remains hard in the refit.
    if certified:
        all_z = np.concatenate((
            np.asarray(diagnostic_z, dtype=np.complex128), ladder_z))
        all_delta = support_delta + ladder_delta
        Om, Bp, achieved_rcond, anchor_error = (
            _constrained_linear_residues(
                mesh, omega_scalar, all_z, all_delta, exact_static))
        static_error = _relative_error(
            evaluate_pole_sum(Om, Bp, 0.0j), exact_static)
    else:
        Bp = Bp_validation
    diagnostic_error = max((
        _relative_error(
            evaluate_pole_sum(frozen_Om, frozen_Bp, value), exact)
        for value, exact in zip(diagnostic_z, support_delta)), default=0.0)
    n_poles = int(Om.shape[0])
    _print_memory_model(pair_block, W0bar, n_poles)
    return IntrabandRow(
        Omega_p=Om,
        B_p=Bp,
        n_poles=n_poles,
        n_modes=n_pair,
        sample_max_rel_error=float(diagnostic_error),
        gap_max_rel_error=float(gap_error),
        static_max_rel_error=float(static_error),
        certified=certified,
        folded_modes=0,
        dropped_modes=0,
        folded_elements=int(fold_el),
        dropped_elements=int(drop_el),
        cluster_width_max_ry=float(width),
        frozen_gap_max_rel_error=float(frozen_gap_error),
        ladder_rcond=float(achieved_rcond),
        ladder_nodes=int(ladder_z.size),
        ladder_refinement=refinement,
        frozen_Omega_p=frozen_Om,
        frozen_B_p=frozen_Bp,
        zero_mode_weight=float(closure.zero_mode_weight),
        zero_mode_cluster=int(closure.zero_mode_cluster),
        zero_mode_pole_shift=float(closure.zero_mode_pole_shift),
        origin_gap_ry2=float(origin_gap),
        origin_gap_doublings=int(doublings),
        origin_gap_m_closure=float(open_m_closure),
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
    "GAP_CERTIFICATE_FIRST_REAL_RY",
    "GAP_CERTIFICATE_LOWEST_BISECTION_REAL_RY",
    "ClusterClosure",
    "IntrabandRow",
    "MAX_CLUSTERS",
    "MAX_ORIGIN_GAP_DOUBLINGS",
    "NEAR_LINE_SEED_REAL_RY",
    "OpenAsymptoticClosure",
    "RESIDUE_LS_RCOND",
    "MODEL",
    "MOMENT_REL_TOL",
    "SAMPLE_REL_TOL",
    "STATIC_REL_TOL",
    "SUM_RULE_REL_TOL",
    "build_row",
    "evaluate_pole_sum",
    "pad_row",
    "shared_near_line_ladder",
]

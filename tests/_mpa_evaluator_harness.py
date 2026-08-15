"""Test oracle for the sampling-plan evaluator.

The ``evaluate_samples`` cluster, moved verbatim out of
``gw.mpa.evaluator``: it has zero production callers and serves only as
the oracle for the shipped-table tests (``tests/test_mpa_evaluator.py``
and ``tests/test_minimax_imag_tables.py``).  Production keeps
``damped_line_rule`` and the ``damped_rectangle_*`` family in
``gw.mpa.evaluator``; this module imports them rather than copying them.
Plain module, no pytest imports.
"""

import numpy as np

import jax
import jax.numpy as jnp

from gw.mpa import sample_plan
from gw.mpa.evaluator import (DEFAULT_REL_TOL,
                              DEFAULT_WAVELENGTHS_PER_PANEL,
                              damped_line_rule)
from gw.minimax_config import MinimaxConfig
from gw.minimax_screening import (build_real_quadrature,
                                  solve_laplace_minimax_imag_interval,
                                  solve_laplace_minimax_interval)

#: Target error handed to the SHIPPED families on the existing-kernel
#: routes.  1e-10 and not tighter because the ``exponential_sum_imag``
#: family has zero shipped tables (DESIGN_minimax R4 / R1) and solves
#: at runtime: 1e-10 returns in milliseconds, 1e-12 measured 96 s.
DEFAULT_KERNEL_TARGET_ERROR = 1.0e-10


def _require_x64():
    """Refuse to run in single precision.  Mirrors ``pade_fit``.

    Every number here is a complex128 resolvent evaluated near its own
    pole; in float32 the strip samples lose the imaginary part that
    makes them well posed.
    """

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "GATE x64_enabled: jax x64 is off, so the evaluator would "
            "run in single precision. FALSE case: "
            "jax.config.update('jax_enable_x64', True) before import, "
            "or JAX_ENABLE_X64=1 in the environment.")


def damped_line_projection(rule, z):
    """The per-point scalar weights ``w_l(z) = -2 h_l e^{i z t_l}``.

    This is the ONLY thing that depends on the sample point: the nodes,
    the weights and the sweep over them are shared by every point on
    the line.  Theory-plan section B's "common damping and common nodes
    per line, scalar per-point projections" is this function returning
    a vector whose only ``z`` dependence is a phase times the line's
    common ``e^{-varpi t}``.
    """

    zc = complex(z)
    if abs(zc.imag - rule["varpi"]) > 1.0e-12 * max(rule["varpi"], 1.0):
        raise ValueError(
            f"GATE projection_on_line: z={z!r} has varpi={zc.imag!r} "
            f"but the rule was built for varpi={rule['varpi']!r}. "
            "FALSE case: the point lies on the line whose rule it is "
            "projected against -- the damping is COMMON to the line "
            "and only the phase is per point.")
    t = jnp.asarray(rule["t"], dtype=jnp.float64)
    h = jnp.asarray(rule["h"], dtype=jnp.float64)
    phase = jnp.exp(1j * jnp.asarray(zc, dtype=jnp.complex128) * t)
    return sample_plan.KERNEL_FACTOR * h.astype(jnp.complex128) * phase


def sine_sweep_from_spectrum(t, delta, weight):
    """``S_l = sum_j g_j sin(Delta_j t_l)``.  The stand-in sweep.

    THIS FUNCTION IS THE SEAM.  In production the per-node quantity is
    a chi0 build at time node ``t_l`` -- ``G(t) G(-t)`` contracted into
    the ISDF basis -- and its spectral content is exactly this sum with
    the transition weights of the band pair.  Here the spectrum is
    materialised because the point is to have an oracle; the
    production evaluator passes its own callable of the same shape
    through ``evaluate_samples(..., sine_sweep_fn=...)`` and nothing
    above this line changes.

    ``weight`` may carry trailing axes -- ``(n_delta,)`` for a scalar
    channel, ``(n_delta, n_chan)`` for a tile of ISDF elements -- and
    the result is ``(n_nodes,) + weight.shape[1:]``.
    """

    tt = jnp.asarray(t, dtype=jnp.float64)
    d = jnp.asarray(delta, dtype=jnp.float64)
    g = jnp.asarray(weight, dtype=jnp.float64)
    if d.ndim != 1 or g.shape[0] != d.shape[0]:
        raise ValueError(
            f"GATE spectrum_shapes: delta has shape {d.shape} and "
            f"weight has shape {g.shape}. FALSE case: delta is "
            "(n_delta,) and weight is (n_delta,) + channel_shape.")
    basis = jnp.sin(tt[:, None] * d[None, :])
    return jnp.tensordot(basis, g, axes=(1, 0))


def exp_sweep_from_spectrum(tau, delta, weight):
    """``E_l = sum_j g_j exp(-tau_l Delta_j)``.  The other seam.

    The existing-kernel routes are exponential sums in ``Delta``, so
    their per-node quantity is a Laplace-domain sweep and not a sine
    one -- see ``evaluate_samples`` for what that difference costs the
    unification.  Same shape contract as
    ``sine_sweep_from_spectrum``.
    """

    tt = jnp.asarray(tau, dtype=jnp.float64)
    d = jnp.asarray(delta, dtype=jnp.float64)
    g = jnp.asarray(weight, dtype=jnp.float64)
    if d.ndim != 1 or g.shape[0] != d.shape[0]:
        raise ValueError(
            f"GATE spectrum_shapes: delta has shape {d.shape} and "
            f"weight has shape {g.shape}. FALSE case: delta is "
            "(n_delta,) and weight is (n_delta,) + channel_shape.")
    basis = jnp.exp(-tt[:, None] * d[None, :])
    return jnp.tensordot(basis, g, axes=(1, 0))


def evaluate_damped_points(rule, z_values, sweep):
    """Project one line's sweep onto its points.  ``(n_z,) + chan``.

    The whole line rides one ``sweep``; each point costs one complex
    scalar reduction over the nodes.
    """

    s = jnp.asarray(sweep)
    out = [jnp.tensordot(damped_line_projection(rule, z), s, axes=(0, 0))
           for z in z_values]
    return jnp.stack(out, axis=0)


# ---------------------------------------------------------------------------
# The existing-kernel routes -- the three cells that already ship
# ---------------------------------------------------------------------------

def existing_kernel_rule(point, *, delta_min, delta_max,
                         target_error=DEFAULT_KERNEL_TARGET_ERROR,
                         max_nodes=64, use_shipped_tables=True):
    """Build the shipped quadrature serving one non-strip point.

    THE CONTRACT ADAPTATION, STATED ONCE.  The shipped families are
    exponential sums IN ``Delta`` -- ``sum_l alpha_l e^{-tau_l
    Delta}`` -- and they are solved on a bounded POSITIVE INTERVAL
    ``[delta_min, delta_max]``, whose ratio ``R`` is the parameter the
    catalog is indexed by.  The damped rule is a sine sum IN ``t`` and
    needs no ``delta_min`` at all, only ``varpi`` and a bandwidth.  So
    the sampling object cannot carry one "domain" field: the three
    shipped cells live on an interval domain and the strip on a
    bandwidth domain, and this function is where the
    interval requirement is actually imposed.  A caller with a gapless
    spectrum (metals, intraband transitions down to zero) has an
    ``R -> inf`` problem on the shipped routes and no problem at all on
    the strip -- which is a reason the protocol puts the metal origin
    sample at ``i*1e-5 Ha`` rather than at 0.

    THE SECOND ADAPTATION, on the ``sine_sum`` cell.  Its shipped
    route (``build_real_quadrature``) decomposes
    ``Delta/(Delta**2-omega**2)`` into two ``1/y`` minimaxes and
    returns SIGNED ``tau``: positive on the ``(omega+Delta)`` branch,
    negative on the ``(omega-Delta)`` branch.  So a "time node" of that
    family is not a time at all, and ``exp_sweep_fn`` must tolerate
    ``exp(+|tau| Delta)``.  It does -- ``|tau| ~ 1/omega`` there, so
    the growing branch is harmless -- but the sampling object cannot
    promise that node vectors are nonnegative, and this is why.

    The returned dict is deliberately the same shape as
    ``damped_line_rule``'s: ``t``/``h`` are the nodes and weights to
    sweep and project, ``n_nodes`` is what the cost report counts. The
    ``KERNEL_FACTOR`` of ``-2`` is NOT folded into ``h`` here; it is
    applied once, by ``evaluate_samples``, for both routes.
    """

    fam = point["family"]
    if fam == "damped_line":
        raise ValueError(
            f"GATE existing_kernel_cell: point {point['role']!r} at "
            f"z={point['z']!r} is a strip sample, which has no shipped "
            "family. FALSE case: the point's family is one of "
            "exponential_sum, exponential_sum_imag, sine_sum -- the "
            "strip goes through damped_line_rule.")
    lo = float(delta_min)
    hi = float(delta_max)
    if not (0.0 < lo < hi) or not np.isfinite(hi):
        raise ValueError(
            f"GATE existing_kernel_interval: (delta_min, delta_max)="
            f"({delta_min!r}, {delta_max!r}) is not a bounded positive "
            "transition interval. FALSE case: 0 < delta_min < "
            "delta_max < inf -- the shipped families are solved on "
            "[delta_min, delta_max] and are indexed by its ratio.")
    if fam == "exponential_sum":
        quad = solve_laplace_minimax_interval(
            lo, hi, target_error=float(target_error),
            max_nodes=int(max_nodes),
            use_shipped_tables=bool(use_shipped_tables))
    elif fam == "exponential_sum_imag":
        # ``varpi`` is a sampling line height over the transition floor,
        # so this request carries the height clause of the beta envelope
        # (GATE0_IMAG_ENVELOPE.md sec 2).  At the fit stage's 1e-12 tier
        # nothing is tabulated yet, so the selector refuses by name and
        # this falls through to the runtime solve exactly as before.
        quad = solve_laplace_minimax_imag_interval(
            lo, hi, point["varpi"], target_error=float(target_error),
            max_nodes=int(max_nodes),
            use_shipped_tables=bool(use_shipped_tables))
    else:
        # sine_sum.  build_real_quadrature shifts an existing 1/y
        # minimax on [delta_min, delta_max] by +/- omega, so the
        # static solve is its input, not a duplicate of it.
        if point["omega"] <= hi:
            raise ValueError(
                f"GATE sine_sum_above_band: point {point['role']!r} "
                f"at z={point['z']!r} has omega={point['omega']!r} at "
                f"or below delta_max={hi!r}. The shipped real-axis "
                "route needs omega above every transition so both 1/y "
                "branches stay positive. FALSE case: omega > "
                "delta_max -- which sample_plan.refuse_unsupported "
                "checks before any physics runs.")
        base = solve_laplace_minimax_interval(
            lo, hi, target_error=float(target_error),
            max_nodes=int(max_nodes),
            use_shipped_tables=bool(use_shipped_tables))
        quad = build_real_quadrature(
            base, point["omega"],
            MinimaxConfig(target_error=float(target_error),
                          max_nodes=int(max_nodes)))
    return {
        "t": np.asarray(quad.tau, dtype=np.float64),
        "h": np.asarray(quad.alpha, dtype=np.float64),
        "n_nodes": int(quad.node_count),
        "family": fam,
        "max_error": float(quad.max_error),
        "delta_min": lo,
        "delta_max": hi,
        "target_error": float(target_error),
    }


# ---------------------------------------------------------------------------
# The per-point evaluation contract
# ---------------------------------------------------------------------------

def evaluate_samples(
    plan,
    delta,
    weight,
    *,
    batching="per-point",
    rel_tol=DEFAULT_REL_TOL,
    wavelengths_per_panel=DEFAULT_WAVELENGTHS_PER_PANEL,
    kernel_target_error=DEFAULT_KERNEL_TARGET_ERROR,
    sine_sweep_fn=sine_sweep_from_spectrum,
    exp_sweep_fn=exp_sweep_from_spectrum,
):
    """Evaluate every point of ``plan`` against a transition spectrum.

    THE SIGNATURE IS THE CONTRACT.  A caller hands over the plan and
    the spectrum and gets back one value per plan point, in plan order,
    plus a cost record.  Theory-plan section B's line-batched
    implementation is a change of ``sine_sweep_fn`` and of
    ``batching``, both of which are arguments; nothing a caller writes
    moves when the sweep stops being a spectrum sum and starts being a
    chi0 build.  That is what "the per-point fallback consumes the
    identical object with a different evaluator" has to mean in code.

    Parameters
    ----------
    plan
        A ``gw.mpa.sample_plan`` plan.  Its ``route`` column decides
        which machine serves each point; nothing here re-derives it.
    delta, weight
        The transition spectrum: ``(n_delta,)`` positive energies and
        ``(n_delta,) + channel_shape`` weights.  ``channel_shape`` is
        free, so a tile of ISDF elements evaluates in one call.
    batching
        ``'per-point'`` (default) gives every strip point its own rule,
        sized to its own ``|omega| + Delta_max`` -- the theory plan's
        named correctness fallback, "independent per-point sweeps".
        ``'per-line'`` builds one rule per line, sized to the line's
        widest point, and rides one sweep across it -- the arithmetic
        of the plan's line-batched design.  The two agree to the
        rule's tolerance; the cost record is where they differ.
    rel_tol, wavelengths_per_panel
        Forwarded to ``damped_line_rule``.
    kernel_target_error
        Forwarded to the shipped families.
    sine_sweep_fn, exp_sweep_fn
        The two seams.  ``sine_sweep_fn(t, delta, weight)`` returns
        ``(n_nodes,) + channel_shape``; ``exp_sweep_fn`` likewise for
        the Laplace nodes.

    Returns
    -------
    ``(values, cost)``
        ``values`` is ``(n_points,) + channel_shape`` complex128 in
        plan order -- feed it straight to ``pade_fit.fit_mpa_poles``
        against ``sample_plan.plan_z(plan)``.  ``cost`` is the record
        ``format_evaluator_cost_report`` prints.
    """

    _require_x64()
    if batching not in ("per-point", "per-line"):
        raise ValueError(
            f"GATE batching_known: batching={batching!r} is not a "
            "known evaluation mode. FALSE case: batching is "
            "'per-point' (the theory plan's correctness fallback) or "
            "'per-line' (its line-batched arithmetic).")
    d = np.asarray(delta, dtype=np.float64)
    if d.ndim != 1 or d.size == 0 or not np.all(d > 0.0):
        raise ValueError(
            f"GATE spectrum_positive: delta has shape {d.shape} and "
            f"min {d.min() if d.size else float('nan')!r}. FALSE case: "
            "delta is a nonempty 1-D array of strictly positive "
            "transition energies.")
    delta_min = float(d.min())
    delta_max = float(d.max())
    sample_plan.refuse_unsupported(plan, delta_max=delta_max)

    points = sample_plan.plan_points(plan)
    routes = sample_plan.plan_routes(plan)
    g = jnp.asarray(weight)
    channel_shape = tuple(np.asarray(weight).shape[1:])
    values = [None] * len(points)

    # --- the three shipped cells: one exponential sweep per point.
    kernel_nodes = 0
    kernel_max_error = 0.0
    for p in routes["existing"]:
        rule = existing_kernel_rule(
            p, delta_min=delta_min, delta_max=delta_max,
            target_error=kernel_target_error)
        swept = exp_sweep_fn(rule["t"], d, g)
        values[p["index"]] = (
            sample_plan.KERNEL_FACTOR
            * jnp.tensordot(jnp.asarray(rule["h"],
                                        dtype=jnp.complex128),
                            jnp.asarray(swept, dtype=jnp.complex128),
                            axes=(0, 0)))
        kernel_nodes += rule["n_nodes"]
        kernel_max_error = max(kernel_max_error, rule["max_error"])

    # --- the strip: the damped-tau rule, shared per line or not.
    damped_nodes = 0
    damped_sweeps = 0
    line_rows = []
    for varpi, line_pts in routes["lines"]:
        if batching == "per-line":
            groups = [line_pts]
        else:
            groups = [(p,) for p in line_pts]
        line_nodes = 0
        worst_kappa0 = 0.0
        order_lo, order_hi = None, None
        for group in groups:
            freq_max = max(abs(p["omega"]) for p in group) + delta_max
            rule = damped_line_rule(
                varpi, freq_max, rel_tol=rel_tol,
                wavelengths_per_panel=wavelengths_per_panel)
            swept = sine_sweep_fn(rule["t"], d, g)
            got = evaluate_damped_points(
                rule, [p["z"] for p in group], swept)
            for k, p in enumerate(group):
                values[p["index"]] = got[k]
            line_nodes += rule["n_nodes"]
            damped_sweeps += 1
            worst_kappa0 = max(worst_kappa0, rule["kappa0"])
            lo_k, hi_k = min(rule["orders"]), max(rule["orders"])
            order_lo = lo_k if order_lo is None else min(order_lo, lo_k)
            order_hi = hi_k if order_hi is None else max(order_hi, hi_k)
        damped_nodes += line_nodes
        line_rows.append({
            "varpi": float(varpi),
            "n_points": len(line_pts),
            "n_sweeps": len(groups),
            "nodes": int(line_nodes),
            "a_dim": float(
                (max(abs(p["omega"]) for p in line_pts) + delta_max)
                / varpi),
            "kappa0": float(worst_kappa0),
            "orders": (int(order_lo), int(order_hi)),
        })

    out = jnp.stack([jnp.asarray(v, dtype=jnp.complex128)
                     for v in values], axis=0)

    # One GN sweep = the static minimax chi0 sweep over the same
    # transition interval at the same tolerance.  It is the unit the
    # theory plan asks the tau-node dispatches to be quoted in.
    gn_quad = solve_laplace_minimax_interval(
        delta_min, delta_max, target_error=kernel_target_error,
        max_nodes=64)
    gn_nodes = int(gn_quad.node_count)
    total_nodes = kernel_nodes + damped_nodes
    cost = {
        "label": plan["label"],
        "census": sample_plan.describe_plan(plan),
        "n_points": len(points),
        "logical_outputs": len(points),
        "channel_shape": channel_shape,
        "batching": batching,
        "delta_interval": (delta_min, delta_max),
        "n_delta": int(d.size),
        "kernel_nodes": int(kernel_nodes),
        "kernel_points": len(routes["existing"]),
        "kernel_max_error": float(kernel_max_error),
        "damped_nodes": int(damped_nodes),
        "damped_points": sum(len(pts) for _, pts in routes["lines"]),
        "damped_sweeps": int(damped_sweeps),
        "lines": tuple(line_rows),
        "node_dispatches": int(total_nodes),
        "gn_sweep_nodes": gn_nodes,
        "gn_sweeps": float(total_nodes) / float(gn_nodes),
        "rq_transforms": len(points),
        "dyson_solves": len(points),
        "rel_tol": float(rel_tol),
        "kernel_target_error": float(kernel_target_error),
    }
    return out, cost


def format_evaluator_cost_report(cost):
    """The evaluation stage's cost, as theory-plan section B demands.

    *"Cost reports must state 2n_p logical outputs, actual tau-node
    dispatches normalized to a GN sweep, and per-output R->q transforms
    and Dyson solves; a line-batched calculation is not 'one build'
    merely because it is one physical sweep."*

    So the node count is printed BESIDE the sweep count, and the
    per-line rows carry both -- ``per-line`` batching turns seven
    sweeps into one without changing the seven R->q transforms and
    seven Dyson solves each of those samples still costs, and the
    report has to make that impossible to misread.
    """

    c = cost
    lines = [
        "",
        "MPA evaluation stage - cost report",
        "-" * 64,
        f"  {c['census']}",
        f"  spectrum        {c['n_delta']} transitions on "
        f"[{c['delta_interval'][0]:.4g}, {c['delta_interval'][1]:.4g}]"
        f", channels {c['channel_shape'] or '()'}",
        f"  batching        {c['batching']}",
        f"  logical outputs {c['logical_outputs']} "
        f"(= 2*n_p samples of W_c)",
        f"  shipped cells   {c['kernel_points']} points, "
        f"{c['kernel_nodes']} nodes, worst family error "
        f"{c['kernel_max_error']:.3e}",
        f"  strip cell      {c['damped_points']} points, "
        f"{c['damped_nodes']} nodes over {c['damped_sweeps']} sweeps "
        f"(rel_tol {c['rel_tol']:.0e})",
    ]
    for row in c["lines"]:
        per_sweep = row["nodes"] / max(row["n_sweeps"], 1)
        lines.append(
            f"    line varpi={row['varpi']:<8.4g} A={row['a_dim']:7.1f} "
            f"{row['n_points']} points / {row['n_sweeps']} sweeps / "
            f"{row['nodes']} nodes "
            f"({per_sweep / max(row['a_dim'], 1e-30):.2f}*A per sweep, "
            f"GL orders {row['orders'][0]}-{row['orders'][1]}, kappa0 "
            f"{row['kappa0']:.4f})")
    lines += [
        f"  node dispatches {c['node_dispatches']} tau nodes "
        f"= {c['gn_sweeps']:.2f} GN sweeps "
        f"(one GN sweep = {c['gn_sweep_nodes']} static minimax nodes "
        f"on the same interval)",
        f"  per output      {c['rq_transforms']} R->q transforms, "
        f"{c['dyson_solves']} Dyson solves "
        "(one each per sample; line batching does not reduce these)",
        "-" * 64,
        "",
    ]
    return "\n".join(lines)

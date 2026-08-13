"""MPA Sigma frequency windows, derived directly from fitted pole geometry.

This module owns scalar planning only.  It reduces each sharded pole field to
the bounds needed by the quadrature fit and returns ordinary ``_SigmaWindow``
objects plus pole selectors.  No pole tile, plan file, or spatial kernel lives
here; execution is the existing GN tau kernel through
``ppm_tau_kernel.get_shared_sigma_tau_kernel``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import minimax

from gw.mpa.evaluator import damped_rectangle_rule
from gw.minimax_screening import MinimaxNodes, solve_phase_minimax_bandwidth
from gw.ppm_windows import (_SigmaBranch, _SigmaWindow,
                            crossing_regularization_floor)


class SharedSigmaWindow(NamedTuple):
    window: _SigmaWindow
    E_A: jax.Array
    omega_abs: np.ndarray
    omega_idx: np.ndarray
    pole_indices: np.ndarray
    bounds: np.ndarray
    phase_real: np.ndarray


_INF = np.inf


def _selector(a_hi=_INF, gamma_lo=-_INF, gamma_hi=_INF):
    # (a_gt, a_le, gamma_ge, gamma_gt, gamma_lt, gamma_le)
    return np.asarray((0.0, a_hi, gamma_lo, -_INF, gamma_hi, _INF),
                      dtype=np.float64)


def _stats(Omega, bounds):
    a, gamma = jnp.real(Omega), -jnp.imag(Omega)
    b = jnp.asarray(bounds)
    mask = ((a > b[0]) & (a <= b[1])
            & (gamma >= b[2]) & (gamma > b[3])
            & (gamma < b[4]) & (gamma <= b[5]))
    live = jnp.abs(Omega) > np.finfo(np.float64).tiny
    count, lo_a, hi_a, lo_g, hi_g, bad = jax.device_get((
        jnp.sum(mask, dtype=jnp.int64),
        jnp.min(jnp.where(mask, a, jnp.inf)),
        jnp.max(jnp.where(mask, a, -jnp.inf)),
        jnp.min(jnp.where(mask, gamma, jnp.inf)),
        jnp.max(jnp.where(mask, gamma, -jnp.inf)),
        jnp.sum(live & ((a <= 0.0) | (gamma < 0.0)), dtype=jnp.int64),
    ))
    if int(bad):
        raise ValueError(
            f"MPA fit contains {int(bad)} unsupported live poles with "
            "Re Omega <= 0 or Im Omega > 0")
    if not int(count):
        return None
    return tuple(map(float, (lo_a, hi_a, lo_g, hi_g)))


def _rows(Omega_poles, bounds, phase_real):
    indices, selectors, phases, stats = [], [], [], []
    for pole, Omega in enumerate(Omega_poles):
        got = _stats(Omega, bounds)
        if got is not None:
            indices.append(pole)
            selectors.append(bounds)
            phases.append(phase_real)
            stats.append(got)
    return (np.asarray(indices, dtype=np.int32),
            np.asarray(selectors, dtype=np.float64).reshape(-1, 6),
            np.asarray(phases, dtype=bool), stats)


def _a_space(branch, predicate):
    E = np.asarray(jax.device_get(branch.E_A), dtype=np.float64)
    base = np.asarray(jax.device_get(branch.base_mask_A), dtype=bool)
    mask = base & predicate(E)
    values = E[mask]
    if not values.size:
        return mask, None
    return mask, (float(np.min(values)), float(np.max(values)))


def _laplace_nodes(rectangles, target_error, max_rank):
    fit = minimax.fit_damped_reciprocal(
        np.asarray(rectangles), target_error=target_error, max_rank=max_rank)
    return (MinimaxNodes(
                t=jnp.asarray(-1j * fit.nodes, dtype=jnp.complex128),
                alpha=jnp.asarray(fit.weights, dtype=jnp.complex128)),
            fit)


def _window(name, nodes, mask_A, E_ref_A, omega_sign, project, prefactor,
            max_error, provenance, *, crossing_kind=None):
    return _SigmaWindow(
        name=name, nodes=nodes, mask_A=mask_A, E_ref_A=E_ref_A,
        E_ref_B=0.0, omega_sign=omega_sign, project=project,
        prefactor=prefactor, crossing_kind=crossing_kind,
        max_error=max_error, provenance=provenance)


def _rectangles(rows, E_bounds, omega_max, *, crossing):
    rectangles = []
    e_lo, e_hi = E_bounds
    for (_a_lo, _a_hi, g_lo, g_hi), real_phase in rows:
        a_lo, a_hi = _a_lo, _a_hi
        x_lo = e_lo + a_lo - (omega_max if crossing else 0.0)
        x_hi = e_hi + a_hi + (0.0 if crossing else omega_max)
        if x_lo <= 0.0:
            raise ValueError(
                "MPA sign-definite window reaches or crosses zero "
                f"(lower denominator {x_lo:.6g} Ry); route this cell "
                "through the crossing family instead")
        rectangles.append((x_lo, x_hi,
                           0.0 if real_phase else g_lo,
                           0.0 if real_phase else g_hi))
    return rectangles


def build_shared_sigma_windows(
    Omega_poles,
    branches: list[_SigmaBranch],
    *,
    regularization_width_ry: float,
    edge_factor: float,
    target_error: float = 1.0e-4,
    max_rank: int = 96,
    hgl_target_error: float = 1.0e-6,
    hgl_eps_q: float = 1.0e-3,
    hgl_max_nodes: int = 200,
    use_shipped_hgl: bool = True,
):
    """Build the complete MPA frequency plan from actual pole bounds.

    ``Gamma < xi`` uses the accepted real-pole HGL functional only in its
    crossing core.  Every sign-definite side retains exact fitted ``Gamma``.
    ``Gamma >= xi`` also retains the complex pole, using a width-certified
    damped real-time core and rotated-Laplace sides.  One sign-definite
    dictionary is shared by all widths and poles in a physical class; memory
    batching is left to the executor.
    """
    poles = tuple(Omega_poles)
    if not poles:
        raise ValueError("MPA Sigma needs at least one pole field")
    omega_max = max((float(np.max(b.omega_abs)) for b in branches
                     if b.omega_abs.size), default=0.0)
    xi = max(float(regularization_width_ry),
             crossing_regularization_floor(omega_max, edge_factor), 1e-12)
    hgl_edge = omega_max + float(edge_factor) * xi

    selectors = {
        "narrow": _selector(gamma_hi=xi),
        "wide": _selector(gamma_lo=xi),
        "wide_shallow": _selector(a_hi=omega_max, gamma_lo=xi),
        "wide_deep": _selector(a_hi=_INF, gamma_lo=xi),
        "narrow_shallow": _selector(a_hi=hgl_edge, gamma_hi=xi),
        "narrow_deep": _selector(a_hi=_INF, gamma_hi=xi),
    }
    # Deep selectors are open at their lower a edge.
    selectors["wide_deep"][0] = omega_max
    selectors["narrow_deep"][0] = hgl_edge
    # Every sign-definite route retains the fitted complex pole.  Replacing
    # Gamma by zero is the HGL core's explicit approximation, not a property
    # of every pole below xi.
    selected = {
        name: _rows(poles, bounds, False)
        for name, bounds in selectors.items()
    }

    output = []
    crossing_branches = []
    sign_specs = []
    for branch in branches:
        crossing = (branch.space == "cond") != branch.neg_omega_half
        neg = -1.0 if branch.neg_omega_half else 1.0
        if not crossing:
            mask, eb = _a_space(branch, lambda E: np.ones(E.shape, bool))
            if eb is not None:
                sign_specs.append((branch, [("single", mask, eb, [
                    ("narrow", selected["narrow"]),
                    ("wide", selected["wide"]),
                ])], -1, -neg))
            continue

        crossing_branches.append((branch, neg))
        mask_slab, eb_slab = _a_space(
            branch, lambda E: np.ones(E.shape, bool))
        if eb_slab is not None:
            sign_specs.append((branch, [("b_slab", mask_slab, eb_slab, [
                ("narrow_deep", selected["narrow_deep"]),
                ("wide_deep", selected["wide_deep"]),
            ])], +1, +neg))
        stripe_outputs = []
        for label, edge, key in (("a_stripe_hgl", hgl_edge,
                                  "narrow_shallow"),
                                 ("a_stripe", omega_max, "wide_shallow")):
            mask, eb = _a_space(branch, lambda E, edge=edge: E > edge)
            if eb is not None:
                stripe_outputs.append((label, mask, eb,
                                       [(key, selected[key])]))
        if stripe_outputs:
            sign_specs.append((branch, stripe_outputs, +1, +neg))

    # Fit each physical sign-definite class once, including both its real-pole
    # and finite-width cells when their A mask is identical.
    for branch, outputs, sign, prefactor in sign_specs:
        rows = []
        for _name, _mask, eb, components in outputs:
            for _key, (_idx, _sel, phase, stats) in components:
                rows.extend((stat, real, eb)
                            for stat, real in zip(stats, phase.tolist()))
        if not rows:
            continue
        rectangles = []
        for stat, real, eb in rows:
            rectangles.extend(_rectangles(
                [(stat, real)], eb, omega_max, crossing=(sign > 0)))
        nodes, fit = _laplace_nodes(
            rectangles, target_error, max_rank)
        for name, mask_A, eb, components in outputs:
            pole_i, bounds, phases = [], [], []
            for _key, (idx, sel, phase, _stats_rows) in components:
                pole_i.extend(idx.tolist())
                bounds.extend(sel.tolist())
                phases.extend(phase.tolist())
            if not pole_i:
                continue
            win = _window(
                name, nodes, mask_A, eb[0], sign, "full", prefactor,
                fit.sampled_max_error,
                f"positive complex-sector fit; sampled error "
                f"{fit.sampled_max_error:.3e}; rank {nodes.t.size}")
            output.append(SharedSigmaWindow(
                win, branch.E_A, branch.omega_abs, branch.omega_idx,
                np.asarray(pole_i, np.int32), np.asarray(bounds),
                np.asarray(phases, bool)))

    if crossing_branches:
        # One finite-width rule serves both causal crossing branches.
        idx, bounds, phase, stats = selected["wide_shallow"]
        exact_masks, f_max = [], 0.0
        if idx.size:
            gamma_min = min(row[2] for row in stats)
            a_span = max(row[1] for row in stats) - min(row[0] for row in stats)
            for branch, neg in crossing_branches:
                mask, eb = _a_space(branch, lambda E: E <= omega_max)
                exact_masks.append((branch, neg, mask, eb))
                if eb is not None:
                    f_max = max(f_max, omega_max + max(omega_max - eb[0], 0.0)
                                + a_span)
            gamma_max = max(row[3] for row in stats)
            rule = damped_rectangle_rule(
                gamma_min, gamma_max, f_max, rel_tol=target_error)
            exact_nodes = MinimaxNodes(
                t=jnp.asarray(rule["t"], dtype=jnp.complex128),
                alpha=jnp.asarray(-1j * rule["h"], dtype=jnp.complex128))
            for _branch, neg, mask, eb in exact_masks:
                if eb is not None:
                    win = _window(
                        "core", exact_nodes, mask, eb[0], +1, "full", -neg,
                        target_error, f"positive damped crossing rule; "
                        f"rank {exact_nodes.t.size}")
                    output.append(SharedSigmaWindow(
                        win, _branch.E_A, _branch.omega_abs,
                        _branch.omega_idx, idx, bounds, phase))

        # The near-axis core is the existing HGL service, not an imitation.
        idx, bounds, _phase, _stats_rows = selected["narrow_shallow"]
        phase = np.ones(idx.size, dtype=bool)
        if idx.size:
            q = solve_phase_minimax_bandwidth(
                max(2.0 * hgl_edge / xi, 1e-8),
                target_error=hgl_target_error, max_nodes=hgl_max_nodes,
                eps_q=hgl_eps_q, target_kind="hgl",
                use_shipped_tables=use_shipped_hgl)
            raw = q.to_minimax_nodes(time_axis="crossing_hgl")
            hgl_nodes = MinimaxNodes(t=raw.t / xi, alpha=raw.alpha / xi)
            for branch, neg in crossing_branches:
                mask, eb = _a_space(branch, lambda E: E <= hgl_edge)
                if eb is not None:
                    win = _window(
                        "core_hgl", hgl_nodes, mask, eb[0], +1, "imag", -neg,
                        q.max_error, q.provenance, crossing_kind="hgl")
                    output.append(SharedSigmaWindow(
                        win, branch.E_A, branch.omega_abs,
                        branch.omega_idx, idx, bounds, phase))

    return output, {"xi_ry": xi, "omega_max_ry": omega_max,
                    "n_windows": len(output),
                    "n_tau": int(sum(row.window.n_tau for row in output))}


__all__ = ["SharedSigmaWindow", "build_shared_sigma_windows"]

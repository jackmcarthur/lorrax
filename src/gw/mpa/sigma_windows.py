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

from gw.mpa.evaluator import damped_rectangle_positive_rule
from gw.minimax_screening import MinimaxNodes
from gw.ppm_windows import _SigmaBranch, _SigmaWindow


class SharedSigmaWindow(NamedTuple):
    window: _SigmaWindow
    E_A: jax.Array
    omega_abs: np.ndarray
    omega_idx: np.ndarray
    pole_indices: np.ndarray
    bounds: np.ndarray
    phase_real: np.ndarray
    #: The owning branch's fractional weight (f or 1−f), or None for the
    #: incumbent bool-mask semantics.  The executor folds it into the A-side
    #: selector operand; planning here uses only the SUPPORT mask.
    band_weight: jax.Array | None = None


_INF = np.inf

#: Node ceiling for the positive causal crossing rule: keeps the validated
#: 500-node safety margin even when mpa_sigma_max_nodes is set lower.
CROSSING_NODE_FLOOR = 500


def _selector(a_hi=_INF, gamma_lo=-_INF, gamma_hi=_INF):
    # (a_gt, a_le, gamma_ge, gamma_gt, gamma_lt, gamma_le)
    return np.asarray((0.0, a_hi, gamma_lo, -_INF, gamma_hi, _INF),
                      dtype=np.float64)


@jax.jit
def _stats_all_poles(Omega, B, bounds):
    """Per-pole selector bounds without iterating a distributed array."""
    a, gamma = jnp.real(Omega), -jnp.imag(Omega)
    b = jnp.asarray(bounds)
    mask = ((jnp.abs(B) > 0.0) & (a > b[0]) & (a <= b[1])
            & (gamma >= b[2]) & (gamma > b[3])
            & (gamma < b[4]) & (gamma <= b[5]))
    axes = tuple(range(1, Omega.ndim))
    return (
        jnp.sum(mask, axis=axes, dtype=jnp.int64),
        jnp.min(jnp.where(mask, a, jnp.inf), axis=axes),
        jnp.max(jnp.where(mask, a, -jnp.inf), axis=axes),
        jnp.min(jnp.where(mask, gamma, jnp.inf), axis=axes),
        jnp.max(jnp.where(mask, gamma, -jnp.inf), axis=axes),
    )


@jax.jit
def _pole_refusal_counts(Omega, B):
    a, gamma = jnp.real(Omega), -jnp.imag(Omega)
    finite_B = jnp.isfinite(jnp.real(B)) & jnp.isfinite(jnp.imag(B))
    live = finite_B & (jnp.abs(B) > 0.0)
    return (
        jnp.sum(~finite_B, dtype=jnp.int64),
        jnp.sum(live & ((a <= 0.0) | (gamma < 0.0)), dtype=jnp.int64),
    )


def _stats_by_pole(Omega, B, bounds):
    arrays = tuple(np.asarray(x) for x in jax.device_get(
        _stats_all_poles(Omega, B, bounds)))
    out = []
    for i in range(int(Omega.shape[0])):
        if not int(arrays[0][i]):
            out.append(None)
        else:
            out.append(tuple(float(x[i]) for x in arrays[1:]))
    return tuple(out)


def _geometry(branches, regularization_width_ry, edge_factor):
    omega_max = max((float(np.max(b.omega_abs)) for b in branches
                     if b.omega_abs.size), default=0.0)
    eta = float(regularization_width_ry)
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("MPA sigma eta must be finite and positive")
    # Fractional occupations give EVERY branch a possible negative-E_A shell
    # (width ~ few×degauss): the crossing branches through their own support,
    # and the statically-sign-definite branches through wrong-side states (an
    # MP1-fractional state above μ still carries weight f in a "val" branch).
    # Deepening the shallow/deep pole edge by the worst excursion across ALL
    # branches keeps every deep-pole rectangle sign-definite — the crossing
    # slab at x_lo = e_lo + a_lo − ω_max > edge·η, and the sd_slab (the
    # wrong-side sliver × deep poles, whose x has the +ω orientation) at
    # x_lo = e_lo + a_lo ≥ edge·η + ω_max — and routes every straddle
    # through a core rule whose f_max bound covers it.  A non-negative
    # support (every normal insulator) contributes zero, so the insulating
    # geometry is unchanged bit-for-bit.
    excursion = 0.0
    for b in branches:
        _mask, eb = _a_space(b, lambda E: np.ones(E.shape, bool))
        if eb is not None:
            excursion = max(excursion, -min(eb[0], 0.0))
    crossing_edge = omega_max + float(edge_factor) * eta + excursion
    selectors = {
        "all": _selector(),
        "shallow": _selector(a_hi=crossing_edge),
        "deep": _selector(),
    }
    selectors["deep"][0] = crossing_edge
    return omega_max, eta, crossing_edge, selectors


def summarize_sigma_poles(
    Omega_poles,
    B_poles,
    branches,
    *,
    regularization_width_ry,
    edge_factor,
    pole_offset=0,
):
    """Reduce one resident pole batch to the scalar planning evidence."""
    _omega_max, _eta, _crossing_edge, selectors = _geometry(
        branches, regularization_width_ry, edge_factor)
    if B_poles.shape != Omega_poles.shape:
        raise ValueError("Omega_poles and B_poles must have identical shapes")
    nonfinite, bad = map(int, jax.device_get(
        _pole_refusal_counts(Omega_poles, B_poles)))
    if nonfinite:
        raise ValueError(f"MPA fit contains {nonfinite} nonfinite residues")
    if bad:
        raise ValueError(
            f"MPA fit contains {bad} unsupported live poles with "
            "Re Omega <= 0 or Im Omega > 0")
    evidence = {
        name: _stats_by_pole(Omega_poles, B_poles, bounds)
        for name, bounds in selectors.items()
    }
    return tuple(
        (int(pole_offset) + local,
         {name: values[local] for name, values in evidence.items()})
        for local in range(int(Omega_poles.shape[0])))


def _rows_from_summaries(summaries, name, bounds, phase_real):
    indices, selectors, phases, stats = [], [], [], []
    for pole, evidence in summaries:
        got = evidence[name]
        if got is None:
            continue
        indices.append(int(pole))
        selectors.append(bounds)
        phases.append(phase_real)
        stats.append(got)
    return (np.asarray(indices, dtype=np.int32),
            np.asarray(selectors, dtype=np.float64).reshape(-1, 6),
            np.asarray(phases, dtype=bool), stats)


def _a_space(branch, predicate):
    E = np.asarray(jax.device_get(branch.E_A), dtype=np.float64)
    base = np.asarray(jax.device_get(branch.base_mask_A), dtype=bool)
    if branch.band_weight is not None:
        # Metallic branches select multiplicatively (mask x weight in the
        # executor), so base_mask_A spans the whole window and exactly-zero
        # weights would widen the geometry with bands that contribute
        # nothing — the -0.53 Ry phantom excursion of the first metallic
        # arm. Geometry sees the exact nonzero support only.
        w = np.asarray(jax.device_get(branch.band_weight), dtype=np.float64)
        base = base & (w != 0.0)
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


def _apply_external_damping(nodes, eta):
    """Insert the requested retarded broadening exactly once."""
    t = jnp.asarray(nodes.t, dtype=jnp.complex128)
    alpha = (jnp.asarray(nodes.alpha, dtype=jnp.complex128)
             * jnp.exp(-float(eta) * t))
    return MinimaxNodes(t=t, alpha=alpha)


def _window(name, nodes, mask_A, E_ref_A, omega_sign, project, prefactor,
            max_error, provenance, *, crossing_kind=None):
    return _SigmaWindow(
        name=name, nodes=nodes, mask_A=mask_A, E_ref_A=E_ref_A,
        E_ref_B=0.0, omega_sign=omega_sign, project=project,
        prefactor=prefactor, crossing_kind=crossing_kind,
        max_error=max_error, provenance=provenance)


def _rectangles(rows, E_bounds, omega_max, eta, *, crossing):
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
                           0.0 if real_phase else g_lo + eta,
                           0.0 if real_phase else g_hi + eta))
    return rectangles


def build_shared_sigma_windows(
    pole_summaries,
    branches: list[_SigmaBranch],
    *,
    regularization_width_ry: float,
    edge_factor: float,
    target_error: float,
    crossing_target_error: float | None = None,
    max_rank: int,
    crossing_max_nodes: int,
):
    """Build the complete MPA frequency plan from summarized pole bounds.

    ``pole_summaries`` is the ``summarize_sigma_poles`` output (any batch
    partition, concatenated in pole order).

    The requested ``regularization_width_ry`` is a literal retarded ``eta``.
    It is inserted once into every denominator by multiplying each time-node
    weight by ``exp(-eta*t)``.  Fitted pole widths remain in ``W(t)``.

    One positive real-time rule serves every pole width in the crossing core.
    The electronic stripe, plasmon slab, and the two inherently sign-definite
    causal branches use the existing rotated-Laplace minimax service.  Pole
    batching is solely an executor memory policy and never a spectral split.

    ``target_error`` and ``crossing_target_error`` both bound the same
    dimensionless relative residual ``|1-d Q(d)|``.  The latter defaults to
    the former for direct callers; production MPA supplies independently
    validated budgets because the two rule families have different
    observable sensitivity.
    """
    sector_error = float(target_error)
    crossing_error = (sector_error if crossing_target_error is None
                      else float(crossing_target_error))
    if not (0.0 < sector_error < 1.0
            and 0.0 < crossing_error < 1.0):
        raise ValueError("MPA Sigma target errors must lie in (0, 1)")

    summaries = tuple(pole_summaries)
    if not summaries:
        raise ValueError("MPA Sigma needs at least one pole summary")
    omega_max, eta, crossing_edge, selectors = _geometry(
        branches, regularization_width_ry, edge_factor)
    selected = {
        name: _rows_from_summaries(summaries, name, bounds, False)
        for name, bounds in selectors.items()
    }

    output = []
    crossing_branches = []
    sliver_branches = []
    sign_specs = []
    # A statically-sign-definite branch keeps the Laplace family only for
    # states whose denominator x = ω + E_A + Ω stays positive with margin.
    # Wrong-side fractional states (signed E_A ≤ sd_edge) genuinely cross:
    # x passes through zero for shallow poles.  They keep THIS family's
    # algebra — omega_sign = −1 (the +ω orientation) and prefactor = −neg —
    # and only change quadrature technology: deep poles through a Laplace
    # slab (sd_slab, sign-definite by the deepened crossing_edge), shallow
    # poles through the damped positive rule (sd_core below), which is the
    # rule built for denominators through zero.  Reclassifying the whole
    # branch into the crossing family is WRONG (measured −8x disagreement
    # with the exact fractional reference): that flips both the ω
    # orientation and the prefactor for every state.
    sd_edge = float(edge_factor) * eta
    for branch in branches:
        crossing = (branch.space == "cond") != branch.neg_omega_half
        neg = -1.0 if branch.neg_omega_half else 1.0
        if not crossing:
            mask, eb = _a_space(branch, lambda E: np.ones(E.shape, bool))
            if eb is None:
                continue
            if branch.band_weight is None or eb[0] > sd_edge:
                sign_specs.append((branch, [("single", mask, eb, [
                    ("all", selected["all"]),
                ])], -1, -neg))
                continue
            mask_hi, eb_hi = _a_space(branch, lambda E: E > sd_edge)
            if eb_hi is not None:
                sign_specs.append((branch, [("single", mask_hi, eb_hi, [
                    ("all", selected["all"]),
                ])], -1, -neg))
            mask_lo, eb_lo = _a_space(branch, lambda E: E <= sd_edge)
            if eb_lo is not None:
                sign_specs.append((branch, [("sd_slab", mask_lo, eb_lo, [
                    ("deep", selected["deep"]),
                ])], -1, -neg))
                sliver_branches.append((branch, neg, mask_lo, eb_lo))
            continue

        crossing_branches.append((branch, neg))
        mask_slab, eb_slab = _a_space(
            branch, lambda E: np.ones(E.shape, bool))
        if eb_slab is not None:
            sign_specs.append((branch, [("b_slab", mask_slab, eb_slab, [
                ("deep", selected["deep"]),
            ])], +1, +neg))
        mask, eb = _a_space(branch, lambda E: E > crossing_edge)
        if eb is not None:
            sign_specs.append((branch, [("a_stripe", mask, eb, [
                ("shallow", selected["shallow"]),
            ])], +1, +neg))

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
                [(stat, real)], eb, omega_max, eta,
                crossing=(sign > 0)))
        nodes, fit = _laplace_nodes(
            rectangles, sector_error, max_rank)
        nodes = _apply_external_damping(nodes, eta)
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
                np.asarray(phases, bool),
                band_weight=branch.band_weight))

    if crossing_branches:
        # One positive causal rule serves the complete fitted-width interval
        # and both causal crossing branches.  Each branch remains a distinct
        # physical G*W contraction because its band space differs.
        idx, bounds, phase, stats = selected["shallow"]
        core_masks, f_max = [], 0.0
        if idx.size:
            gamma_min = eta + min(row[2] for row in stats)
            gamma_max = eta + max(row[3] for row in stats)
            a_lo = min(row[0] for row in stats)
            a_hi = max(row[1] for row in stats)
            for branch, neg in crossing_branches:
                mask, eb = _a_space(branch, lambda E: E <= crossing_edge)
                core_masks.append((branch, neg, mask, eb))
                if eb is not None:
                    w_lo = float(np.min(branch.omega_abs))
                    w_hi = float(np.max(branch.omega_abs))
                    f_max = max(
                        f_max,
                        *(abs(w - e - a)
                          for w in (w_lo, w_hi)
                          for e in eb
                          for a in (a_lo, a_hi)))
            rule = damped_rectangle_positive_rule(
                gamma_min, gamma_max, f_max, rel_tol=crossing_error,
                max_nodes=crossing_max_nodes)
            raw_nodes = MinimaxNodes(
                t=jnp.asarray(rule["t"], dtype=jnp.complex128),
                alpha=jnp.asarray(-1j * rule["h"], dtype=jnp.complex128))
            exact_nodes = _apply_external_damping(raw_nodes, eta)
            for _branch, neg, mask, eb in core_masks:
                if eb is not None:
                    win = _window(
                        "core", exact_nodes, mask, eb[0], +1, "full", -neg,
                        rule["sampled_max_error"],
                        f"positive {rule['rule_type']} damped crossing rule; "
                        f"eta {eta:.6e}; gamma "
                        f"[{gamma_min:.6e},{gamma_max:.6e}]; "
                        f"f_max {f_max:.6e}; target {crossing_error:.3e}; "
                        f"sampled error {rule['sampled_max_error']:.3e}; "
                        f"continuum cover "
                        f"{rule['continuum_error_bound']:.3e}; "
                        f"tail fraction "
                        f"{rule['tail_budget_fraction']:.3f}; "
                        f"kappa {rule['kappa0']:.6f}; "
                        f"rank {exact_nodes.t.size}")
                    output.append(SharedSigmaWindow(
                        win, _branch.E_A, _branch.omega_abs,
                        _branch.omega_idx, idx, bounds, phase,
                        band_weight=_branch.band_weight))

    if sliver_branches:
        # Wrong-side slivers of the sign-definite branches × shallow poles:
        # the same damped positive rule as the crossing core (it covers
        # |Re x| ≤ f_max through zero; the retarded decay comes from the
        # pole widths + eta, not the sign of Re x), but evaluated in THIS
        # family's orientation: x = ω + E_A + Ω, i.e. omega_sign = −1 and
        # prefactor = −neg.  f_max therefore bounds |ω + e + a|, not
        # |ω − e − a|.
        idx, bounds, phase, stats = selected["shallow"]
        if idx.size:
            gamma_min = eta + min(row[2] for row in stats)
            gamma_max = eta + max(row[3] for row in stats)
            a_lo = min(row[0] for row in stats)
            a_hi = max(row[1] for row in stats)
            for branch, neg, mask_lo, eb_lo in sliver_branches:
                w_lo = float(np.min(branch.omega_abs))
                w_hi = float(np.max(branch.omega_abs))
                f_max = max(abs(w + e + a)
                            for w in (w_lo, w_hi)
                            for e in eb_lo
                            for a in (a_lo, a_hi))
                rule = damped_rectangle_positive_rule(
                    gamma_min, gamma_max, f_max, rel_tol=crossing_error,
                    max_nodes=crossing_max_nodes)
                raw_nodes = MinimaxNodes(
                    t=jnp.asarray(rule["t"], dtype=jnp.complex128),
                    alpha=jnp.asarray(-1j * rule["h"], dtype=jnp.complex128))
                exact_nodes = _apply_external_damping(raw_nodes, eta)
                # Prefactor +neg, NOT −neg: the Laplace service represents
                # +1/x while this damped rule represents −1/x (alpha = −i·h
                # against ∫e^{−ixt}dt = −i/x), which is also why the
                # crossing core flips sign relative to its slab.  The
                # family value −neg/x' therefore needs pref = +neg here.
                win = _window(
                    "sd_core", exact_nodes, mask_lo, eb_lo[0], -1, "full",
                    +neg, rule["sampled_max_error"],
                    f"positive {rule['rule_type']} damped rule in the "
                    f"sign-definite (+ω) orientation; eta {eta:.6e}; gamma "
                    f"[{gamma_min:.6e},{gamma_max:.6e}]; "
                    f"f_max {f_max:.6e}; target {crossing_error:.3e}; "
                    f"sampled error {rule['sampled_max_error']:.3e}; "
                    f"rank {exact_nodes.t.size}")
                output.append(SharedSigmaWindow(
                    win, branch.E_A, branch.omega_abs, branch.omega_idx,
                    idx, bounds, phase,
                    band_weight=branch.band_weight))

    return output, {"eta_ry": eta, "omega_max_ry": omega_max,
                    "crossing_edge_ry": crossing_edge,
                    "sector_target_error": sector_error,
                    "crossing_target_error": crossing_error,
                    "n_windows": len(output),
                    "n_tau": int(sum(row.window.n_tau for row in output))}


__all__ = [
    "CROSSING_NODE_FLOOR", "SharedSigmaWindow",
    "build_shared_sigma_windows", "summarize_sigma_poles",
]

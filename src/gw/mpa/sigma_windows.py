"""MPA Sigma frequency windows, derived directly from fitted pole geometry.

This module owns scalar planning only.  It reduces each sharded pole field to
the bounds needed by the quadrature fit and returns ordinary ``_SigmaWindow``
objects plus pole selectors.  No pole tile, plan file, or spatial kernel lives
here; execution is the existing GN tau kernel through
``ppm_tau_kernel.get_shared_sigma_tau_kernel``.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import minimax

from gw.efermi import (OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
                       band_in_occupation_window, occupation_weight_floor)
from gw.mpa.evaluator import damped_rectangle_positive_rule
from gw.minimax_screening import MinimaxNodes
from gw.ppm_windows import _SigmaBranch, _SigmaWindow, _omega_clusters


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

#: ``occupation_window_threshold`` lives in ``gw.efermi`` — ONE default, ONE
#: occupancy→weight map, ONE predicate, shared with ``gw.ppm_windows`` (the
#: Σ branch supports) and ``gw.w_isdf`` (the χ₀ occupation supports).  Re-bound
#: here because this module's callers and tests have imported these names since
#: the rule was introduced.
_weight_floor = occupation_weight_floor


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


def _geometry(branches, regularization_width_ry, edge_factor, weight_floor):
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
        _mask, eb = _a_space(b, lambda E: np.ones(E.shape, bool),
                             weight_floor)
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
    occupation_window_threshold=OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
):
    """Reduce one resident pole batch to the scalar planning evidence.

    ``occupation_window_threshold`` MUST match the value handed to
    :func:`build_shared_sigma_windows` — the two share ``_geometry``, and a
    mismatch would select poles against one support and windows against
    another.  Both come from the single deck key in production.
    """
    _omega_max, _eta, _crossing_edge, selectors = _geometry(
        branches, regularization_width_ry, edge_factor,
        _weight_floor(occupation_window_threshold))
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


def _a_space(branch, predicate, weight_floor=0.0):
    E = np.asarray(jax.device_get(branch.E_A), dtype=np.float64)
    base = np.asarray(jax.device_get(branch.base_mask_A), dtype=bool)
    if branch.band_weight is not None:
        # Metallic branches select multiplicatively (mask x weight in the
        # executor), so base_mask_A spans the whole window and negligible
        # weights would widen the geometry with bands that contribute
        # nothing — the -0.53 Ry phantom excursion of the first metallic
        # arm, whose val branch reached out to a smallest live weight of
        # 2.67e-322 (a subnormal) because the cut was the EXACT `w != 0.0`.
        # An exact cut only excludes what underflowed to zero, which is
        # ~54 smearing widths out; the occupancy threshold cuts at the
        # physical few-widths shell instead (0.995 ⇒ |w| > 0.005 ⇒ about
        # 4.3 widths).  Exact zeros are still excluded, since 0 is not
        # > 0.005, so that history is preserved, not traded away.
        # The magnitude/never-clipped argument is at
        # ``gw.efermi.band_in_occupation_window``, which owns the predicate.
        #
        # WIDENING THIS COSTS GEOMETRY, NOT JUST BANDS: the support's
        # min(E_A) sets `excursion` in _geometry, which deepens
        # crossing_edge for EVERY branch and moves work between the
        # sign-definite and crossing quadrature families.  That is the
        # mechanism by which an over-wide support refused outright at
        # -0.53 Ry.  Lower the threshold only with a plan-level A/B.
        #
        # REDUNDANT-BY-DESIGN since the same floor is applied to the base
        # masks at ``ppm_windows.branches_for_omega_grid``: this is the cut
        # that OWNS the geometry, and it stays here so a branch built by any
        # other route cannot plan a support the executor does not honour.
        # Idempotent — same floor, same magnitude rule.
        w = np.asarray(jax.device_get(branch.band_weight), dtype=np.float64)
        base = base & band_in_occupation_window(w, float(weight_floor))
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
            max_error, provenance, *, crossing_kind=None, e_ref_b=0.0):
    return _SigmaWindow(
        name=name, nodes=nodes, mask_A=mask_A, E_ref_A=E_ref_A,
        E_ref_B=e_ref_b, omega_sign=omega_sign, project=project,
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


#: Shell rules are cached on this f_max lattice (Ry).  A rule certified at
#: the padded bandwidth covers every smaller request, so rounding UP is
#: free accuracy-wise and turns near-identical per-cluster requests into
#: one rule build.
_F_MAX_LATTICE_RY = 0.25


def _damped_rule_cached(cache, gamma_min, gamma_max, f_max, rel_tol,
                        max_nodes):
    f_pad = _F_MAX_LATTICE_RY * float(
        np.ceil(max(float(f_max), _F_MAX_LATTICE_RY) / _F_MAX_LATTICE_RY))
    # Widths select the target rectangle; unlike the conservative f_max
    # lattice below, rounding them aliases distinct functions.
    key = (float(gamma_min), float(gamma_max),
           f_pad, float(rel_tol), int(max_nodes))
    if key not in cache:
        cache[key] = damped_rectangle_positive_rule(
            gamma_min, gamma_max, f_pad, rel_tol=rel_tol,
            max_nodes=max_nodes)
    return cache[key]


def _damped_nodes(rule, eta):
    """Executor placement of a positive damped rule: α = −i·h, real t."""
    return _apply_external_damping(
        MinimaxNodes(
            t=jnp.asarray(rule["t"], dtype=jnp.complex128),
            alpha=jnp.asarray(-1j * rule["h"], dtype=jnp.complex128)),
        eta)


def _stats_rectangles(stats, phases, x_lo_of, x_hi_of, eta):
    """One rectangle per shallow pole row for a sign-definite bulk cell.

    ``x_lo_of(a_lo, a_hi)`` / ``x_hi_of(a_lo, a_hi)`` map a row's pole
    bracket to the cell's denominator bounds.  Same γ semantics as
    ``_rectangles``: fitted width + η for complex rows, 0 for real rows
    (η enters at execution through ``_apply_external_damping``).
    """
    rectangles = []
    for (a_lo, a_hi, g_lo, g_hi), real_phase in zip(stats, phases):
        x_lo = x_lo_of(a_lo, a_hi)
        x_hi = x_hi_of(a_lo, a_hi)
        if x_lo <= 0.0:
            raise ValueError(
                "GATE bulk_cell_sign_definite: a clustered bulk cell "
                f"reaches or crosses zero (lower denominator {x_lo:.6g} "
                "Ry).  FALSE case: every bulk rectangle keeps the "
                "shell margin — this cell belongs to the shell rule.")
        rectangles.append((x_lo, x_hi,
                           0.0 if real_phase else g_lo + eta,
                           0.0 if real_phase else g_hi + eta))
    return rectangles


def _conjugate_laplace_nodes(rectangles, target_error, max_rank):
    """Rotated-Laplace nodes for denominators with POSITIVE Im part.

    ``fit_damped_reciprocal`` represents ``1/(x − iγ)`` on lower-half
    rectangles.  The retarded pos-slab denominator ``ω − e − a + iγ_tot``
    is its complex conjugate, so the executed rule is the conjugated
    fit: ``1/conj(d) = Σ conj(w) exp(−conj(d)·conj(n))``, which in the
    executor's ``exp(−i·x·t)`` placement is ``t = +i·conj(n)`` — the
    mirror of the sign-definite family's ``t = −i·n``.
    """
    fit = minimax.fit_damped_reciprocal(
        np.asarray(rectangles), target_error=target_error,
        max_rank=max_rank)
    return (MinimaxNodes(
                t=jnp.asarray(1j * np.conj(fit.nodes),
                              dtype=jnp.complex128),
                alpha=jnp.asarray(np.conj(fit.weights),
                                  dtype=jnp.complex128)),
            fit)


#: Refusal ceiling on any single factored exponent of a pos-slab window.
#: float64 overflows at exp(~709); past ~600 the certified product still
#: fits but headroom for the residue magnitudes is gone.
_POS_SLAB_LOG_CAP = 600.0


def _pos_slab_overflow_gate(nodes, gamma_hi_tot):
    """Refuse a pos-slab whose factored W term could overflow.

    With ``Im t > 0`` the kernel's factorised exponentials grow: the
    references are anchored so the band, pole-energy, and ω factors all
    decay in-window, leaving one growing term — the pole-width part of
    W, ``exp(γ·Im n)`` with ``Im n = Re t``.  The fit certifies the
    assembled residual, so growth is legitimate; overflow is not.
    """
    grow = float(gamma_hi_tot) * float(np.max(np.real(
        np.asarray(nodes.t))))
    if grow > _POS_SLAB_LOG_CAP:
        raise ValueError(
            "GATE pos_slab_exponent: the conjugate-placement bulk "
            f"window carries a factored exponent {grow:.1f} above the "
            f"overflow cap {_POS_SLAB_LOG_CAP:.0f} (gamma_max+eta = "
            f"{gamma_hi_tot:.3e} Ry x max Im node).  FALSE case: pole "
            "widths x fit-node depth stay under the float64 range — "
            "loosen the sector target error or narrow the pole widths.")
    return grow


def build_shared_sigma_windows(
    pole_summaries,
    branches: list[_SigmaBranch],
    *,
    regularization_width_ry: float,
    edge_factor: float,
    target_error: float,
    max_rank: int,
    crossing_max_nodes: int,
    omega_grid_step_ry: float,
    occupation_window_threshold: float = OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
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

    ``target_error`` bounds the dimensionless residual ``|1-d Q(d)|``.
    The delivered planner apportions it among windows by delivered mass.

    Omega clusters come from the requested grid and its own step
    (``sigma_omega_step_ev``), not from a separate tolerance.  Every patch in
    ``sigma_omega_patches_ev`` is built AT that step, so consecutive samples
    within a patch sit exactly one step apart and only a patch boundary can
    exceed it; a break wider than 1.5 steps is therefore a genuinely missing
    patch.  Patched semicore grids split, contiguous grids stay monolithic
    bit-for-bit, and a patch holding a single sample is still resolved — which
    is why the step is read from the deck rather than inferred from spacings.

    ``occupation_window_threshold`` is the OCCUPANCY at which a band leaves a
    Green's-function branch; :func:`_weight_floor` converts it to the branch
    weight floor ``1 - threshold`` that :func:`_a_space` applies.  It only
    reaches branches carrying a ``band_weight`` (metals); an insulating plan
    has ``band_weight=None`` and is untouched, bit-for-bit.  It MUST equal
    the value given to :func:`summarize_sigma_poles`.
    """
    sector_error = float(target_error)
    crossing_error = sector_error
    if not 0.0 < sector_error < 1.0:
        raise ValueError("MPA Sigma target error must lie in (0, 1)")

    summaries = tuple(pole_summaries)
    if not summaries:
        raise ValueError("MPA Sigma needs at least one pole summary")
    # ONE floor for the whole plan: bind it once so no support below can
    # silently plan against a different band set than _geometry measured.
    weight_floor = _weight_floor(occupation_window_threshold)
    a_space = functools.partial(_a_space, weight_floor=weight_floor)
    omega_max, eta, crossing_edge, selectors = _geometry(
        branches, regularization_width_ry, edge_factor, weight_floor)
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
            mask, eb = a_space(branch, lambda E: np.ones(E.shape, bool))
            if eb is None:
                continue
            if branch.band_weight is None or eb[0] > sd_edge:
                sign_specs.append((branch, [("single", mask, eb, [
                    ("all", selected["all"]),
                ])], -1, -neg))
                continue
            mask_hi, eb_hi = a_space(branch, lambda E: E > sd_edge)
            if eb_hi is not None:
                sign_specs.append((branch, [("single", mask_hi, eb_hi, [
                    ("all", selected["all"]),
                ])], -1, -neg))
            mask_lo, eb_lo = a_space(branch, lambda E: E <= sd_edge)
            if eb_lo is not None:
                sign_specs.append((branch, [("sd_slab", mask_lo, eb_lo, [
                    ("deep", selected["deep"]),
                ])], -1, -neg))
                sliver_branches.append((branch, neg, mask_lo, eb_lo))
            continue

        crossing_branches.append((branch, neg))
        mask_slab, eb_slab = a_space(
            branch, lambda E: np.ones(E.shape, bool))
        if eb_slab is not None:
            sign_specs.append((branch, [("b_slab", mask_slab, eb_slab, [
                ("deep", selected["deep"]),
            ])], +1, +neg))
        mask, eb = a_space(branch, lambda E: E > crossing_edge)
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

    gap_ry = 1.5 * float(omega_grid_step_ry)
    if not (np.isfinite(gap_ry) and gap_ry > 0.0):
        raise ValueError(
            "MPA sigma omega grid step must be finite and positive")
    #: The same protection the crossing_edge carries: edge margin + the
    #: worst wrong-side excursion across all branches.
    margin = crossing_edge - omega_max
    rule_cache = {}

    if crossing_branches:
        # One positive causal rule per (branch, ω cluster) shell.  On a
        # contiguous ω grid every branch has one cluster and the incumbent
        # monolithic geometry below is reproduced bit-for-bit; on a gapped
        # (patched) grid the core decomposes per cluster into a shell rule
        # whose bandwidth is set by the cluster span and pole bracket —
        # independent of the dynamic range — plus sign-definite bulk slabs
        # on the logarithmic rotated-Laplace family
        # (docs/dev/crossing-rule-cost-law.md).
        idx, bounds, phase, stats = selected["shallow"]
        if idx.size:
            gamma_min = eta + min(row[2] for row in stats)
            gamma_max = eta + max(row[3] for row in stats)
            a_lo = min(row[0] for row in stats)
            a_hi = max(row[1] for row in stats)
            branch_clusters = [
                (branch, neg, _omega_clusters(branch.omega_abs, gap_ry))
                for branch, neg in crossing_branches]
            monolithic = all(
                len(clusters) == 1 for _b, _n, clusters in branch_clusters)
        if idx.size and monolithic:
            core_masks, f_max = [], 0.0
            for branch, neg in crossing_branches:
                mask, eb = a_space(branch, lambda E: E <= crossing_edge)
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
            exact_nodes = _damped_nodes(rule, eta)
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
        elif idx.size:
            for _branch, neg, clusters in branch_clusters:
                mask_core, eb_core = a_space(
                    _branch, lambda E: E <= crossing_edge)
                if eb_core is None:
                    continue
                omega_all = np.asarray(_branch.omega_abs, np.float64)
                idx_all = np.asarray(_branch.omega_idx)
                for om_sel, w_lo, w_hi in clusters:
                    omega_c = omega_all[om_sel]
                    omega_idx_c = idx_all[om_sel]
                    e_lo_split = w_lo - a_hi - margin
                    e_hi_split = w_hi - a_lo + margin
                    mask_p, eb_p = a_space(
                        _branch,
                        lambda E: (E < e_lo_split) & (E <= crossing_edge))
                    mask_s, eb_s = a_space(
                        _branch,
                        lambda E: (E >= e_lo_split) & (E <= e_hi_split)
                        & (E <= crossing_edge))
                    mask_n, eb_n = a_space(
                        _branch,
                        lambda E: (E > e_hi_split) & (E <= crossing_edge))
                    if not np.array_equal(
                            mask_p | mask_s | mask_n, mask_core):
                        raise AssertionError(
                            "GATE core_cluster_partition: the three "
                            "clustered core cells do not tile the core "
                            "support exactly — a band was lost or "
                            "double-counted.")
                    phases_rows = phase.tolist()
                    if eb_s is not None:
                        # Poles above a_cut cannot cross THIS cluster for
                        # any actual band (e >= -excursion), so they leave
                        # the eta-resolved shell for a sign-definite slab
                        # below.  Without this cut the shell bandwidth is
                        # the WHOLE shallow-pole spatial spread (~5 Ry on
                        # the sodium store) and the decomposition loses to
                        # the monolithic rule.
                        a_cut = w_hi + margin
                        keep = [i for i, row in enumerate(stats)
                                if row[0] <= a_cut]
                        a_hi_s = min(a_hi, a_cut)
                        f_c = max(abs(w - e - a)
                                  for w in (w_lo, w_hi)
                                  for e in eb_s
                                  for a in (a_lo, a_hi_s))
                        rows_bounds = np.asarray(bounds)[keep].copy()
                        rows_bounds[:, 1] = np.minimum(
                            rows_bounds[:, 1], a_cut)
                        rule = _damped_rule_cached(
                            rule_cache, gamma_min, gamma_max, f_c,
                            crossing_error, crossing_max_nodes)
                        win = _window(
                            "core", _damped_nodes(rule, eta), mask_s,
                            eb_s[0], +1, "full", -neg,
                            rule["sampled_max_error"],
                            f"positive {rule['rule_type']} damped crossing "
                            f"shell; omega cluster "
                            f"[{w_lo:.6e},{w_hi:.6e}]; eta {eta:.6e}; "
                            f"gamma [{gamma_min:.6e},{gamma_max:.6e}]; "
                            f"a <= {a_cut:.6e}; "
                            f"f_max {rule['freq_max']:.6e}; target "
                            f"{crossing_error:.3e}; sampled error "
                            f"{rule['sampled_max_error']:.3e}; "
                            f"kappa {rule['kappa0']:.6f}; "
                            f"rank {rule['n_nodes']}")
                        output.append(SharedSigmaWindow(
                            win, _branch.E_A, omega_c, omega_idx_c,
                            np.asarray(idx)[keep], rows_bounds,
                            np.asarray(phase)[keep],
                            band_weight=_branch.band_weight))
                        deep_rows = [i for i, row in enumerate(stats)
                                     if row[1] > a_cut]
                        if deep_rows:
                            kept_stats = [stats[i] for i in deep_rows]
                            kept_phase = np.asarray(phase)[deep_rows]
                            rectangles = _stats_rectangles(
                                kept_stats, kept_phase.tolist(),
                                lambda r_lo, _r_hi:
                                    eb_s[0] + max(r_lo, a_cut) - w_hi,
                                lambda _r_lo, r_hi:
                                    eb_s[1] + r_hi - w_lo,
                                eta)
                            nodes, fit = _laplace_nodes(
                                rectangles, sector_error, max_rank)
                            nodes = _apply_external_damping(nodes, eta)
                            deep_bounds = np.asarray(bounds)[
                                deep_rows].copy()
                            deep_bounds[:, 0] = np.maximum(
                                deep_bounds[:, 0], a_cut)
                            win = _window(
                                "c_neg_slab", nodes, mask_s, eb_s[0], +1,
                                "full", +neg, fit.sampled_max_error,
                                f"uncrossable deep-pole side of the shell "
                                f"(a > {a_cut:.6e}); omega cluster "
                                f"[{w_lo:.6e},{w_hi:.6e}]; sampled error "
                                f"{fit.sampled_max_error:.3e}; rank "
                                f"{nodes.t.size}",
                                e_ref_b=a_cut)
                            E_A_local = jnp.where(
                                jnp.reshape(jnp.asarray(mask_s),
                                            _branch.E_A.shape),
                                _branch.E_A, eb_s[0])
                            output.append(SharedSigmaWindow(
                                win, E_A_local, omega_c, omega_idx_c,
                                np.asarray(idx)[deep_rows], deep_bounds,
                                kept_phase,
                                band_weight=_branch.band_weight))
                    if eb_n is not None:
                        rectangles = _stats_rectangles(
                            stats, phases_rows,
                            lambda r_lo, _r_hi: eb_n[0] + r_lo - w_hi,
                            lambda _r_lo, r_hi: eb_n[1] + r_hi - w_lo,
                            eta)
                        nodes, fit = _laplace_nodes(
                            rectangles, sector_error, max_rank)
                        nodes = _apply_external_damping(nodes, eta)
                        win = _window(
                            "c_neg_slab", nodes, mask_n, eb_n[0], +1,
                            "full", +neg, fit.sampled_max_error,
                            f"clustered sign-definite bulk (x = e+a-omega "
                            f">= {margin:.3e}); omega cluster "
                            f"[{w_lo:.6e},{w_hi:.6e}]; sampled error "
                            f"{fit.sampled_max_error:.3e}; rank "
                            f"{nodes.t.size}",
                            e_ref_b=a_lo)
                        # Masked-out SHALLOWER bands grow as
                        # exp((E_ref-E)·n) under decaying nodes; the
                        # metallic float selector multiplies rather than
                        # where-selects, so clamp them to the reference
                        # (exact zeros instead of 0·inf).
                        E_A_local = jnp.where(
                            jnp.reshape(jnp.asarray(mask_n),
                                        _branch.E_A.shape),
                            _branch.E_A, eb_n[0])
                        output.append(SharedSigmaWindow(
                            win, E_A_local, omega_c, omega_idx_c,
                            idx, bounds, phase,
                            band_weight=_branch.band_weight))
                    if eb_p is not None:
                        rectangles = _stats_rectangles(
                            stats, phases_rows,
                            lambda _r_lo, r_hi: w_lo - eb_p[1] - r_hi,
                            lambda r_lo, _r_hi: w_hi - eb_p[0] - r_lo,
                            eta)
                        nodes, fit = _conjugate_laplace_nodes(
                            rectangles, sector_error, max_rank)
                        nodes = _apply_external_damping(nodes, eta)
                        grow = _pos_slab_overflow_gate(
                            nodes, gamma_max)
                        win = _window(
                            "c_pos_slab", nodes, mask_p, eb_p[1], +1,
                            "full", -neg, fit.sampled_max_error,
                            f"clustered sign-definite bulk (x = "
                            f"omega-e-a >= {margin:.3e}, conjugate "
                            f"placement); omega cluster "
                            f"[{w_lo:.6e},{w_hi:.6e}]; sampled error "
                            f"{fit.sampled_max_error:.3e}; max factored "
                            f"log-growth {grow:.1f}; rank {nodes.t.size}",
                            e_ref_b=a_hi)
                        E_A_local = jnp.where(
                            jnp.reshape(jnp.asarray(mask_p),
                                        _branch.E_A.shape),
                            _branch.E_A, eb_p[1])
                        output.append(SharedSigmaWindow(
                            win, E_A_local, omega_c, omega_idx_c,
                            idx, bounds, phase,
                            band_weight=_branch.band_weight))

    if sliver_branches:
        # Wrong-side slivers of the sign-definite branches × shallow poles:
        # the same damped positive rule as the crossing core (it covers
        # |Re x| ≤ f_max through zero; the retarded decay comes from the
        # pole widths + eta, not the sign of Re x), but evaluated in THIS
        # family's orientation: x = ω + E_A + Ω, i.e. omega_sign = −1 and
        # prefactor = −neg.  f_max therefore bounds |ω + e + a|, not
        # |ω − e − a|.
        #
        # On a gapped (multi-cluster) ω grid the sliver decomposes too:
        # x = ω + e + a can only cross zero where BOTH ω and a are within
        # the excursion scale (e ≥ eb_lo[0] ≥ −excursion), so everything
        # with ω + eb_lo[0] + a ≥ edge·η is a sign-definite Laplace cell
        # in this family's own +1/x orientation, and only the tiny
        # (small-ω × small-a) corner keeps a damped rule — with a
        # bandwidth set by the corner, not by ω_max.
        idx, bounds, phase, stats = selected["shallow"]
        if idx.size:
            gamma_min = eta + min(row[2] for row in stats)
            gamma_max = eta + max(row[3] for row in stats)
            a_lo = min(row[0] for row in stats)
            a_hi = max(row[1] for row in stats)

            def _sliver_damped(branch, neg, mask_lo, eb_lo, omega_c,
                               omega_idx_c, f_max, rows_idx, rows_bounds,
                               rows_phase, note, *, pad):
                # ``pad=False`` is the single-cluster (incumbent) path: the
                # raw builder at the exact f_max, bit-for-bit the plan the
                # monolithic geometry always produced.
                if pad:
                    rule = _damped_rule_cached(
                        rule_cache, gamma_min, gamma_max, f_max,
                        crossing_error, crossing_max_nodes)
                else:
                    rule = damped_rectangle_positive_rule(
                        gamma_min, gamma_max, f_max,
                        rel_tol=crossing_error,
                        max_nodes=crossing_max_nodes)
                # Prefactor +neg, NOT −neg: the Laplace service represents
                # +1/x while this damped rule represents −1/x (alpha =
                # −i·h against ∫e^{−ixt}dt = −i/x), which is also why the
                # crossing core flips sign relative to its slab.  The
                # family value −neg/x' therefore needs pref = +neg here.
                win = _window(
                    "sd_core", _damped_nodes(rule, eta), mask_lo,
                    eb_lo[0], -1, "full", +neg, rule["sampled_max_error"],
                    f"positive {rule['rule_type']} damped rule in the "
                    f"sign-definite (+ω) orientation{note}; eta "
                    f"{eta:.6e}; gamma [{gamma_min:.6e},{gamma_max:.6e}]; "
                    f"f_max {rule['freq_max']:.6e}; target "
                    f"{crossing_error:.3e}; sampled error "
                    f"{rule['sampled_max_error']:.3e}; "
                    f"rank {rule['n_nodes']}")
                output.append(SharedSigmaWindow(
                    win, branch.E_A, omega_c, omega_idx_c,
                    rows_idx, rows_bounds, rows_phase,
                    band_weight=branch.band_weight))

            def _sliver_laplace(branch, neg, mask_lo, eb_lo, omega_c,
                                omega_idx_c, w_lo, w_hi, a_floor, note):
                keep = [i for i, row in enumerate(stats)
                        if row[1] > a_floor]
                if not keep:
                    return
                kept_stats = [stats[i] for i in keep]
                rows_idx = np.asarray(idx)[keep]
                rows_bounds = np.asarray(bounds)[keep].copy()
                rows_bounds[:, 0] = np.maximum(rows_bounds[:, 0], a_floor)
                rows_phase = np.asarray(phase)[keep]
                rectangles = _stats_rectangles(
                    kept_stats, rows_phase.tolist(),
                    lambda r_lo, _r_hi:
                        w_lo + eb_lo[0] + max(r_lo, a_floor),
                    lambda _r_lo, r_hi: w_hi + eb_lo[1] + r_hi,
                    eta)
                nodes, fit = _laplace_nodes(
                    rectangles, sector_error, max_rank)
                nodes = _apply_external_damping(nodes, eta)
                win = _window(
                    "sd_shallow_slab", nodes, mask_lo, eb_lo[0], -1,
                    "full", -neg, fit.sampled_max_error,
                    f"sliver sign-definite Laplace cell{note}; omega "
                    f"[{w_lo:.6e},{w_hi:.6e}]; a > {a_floor:.6e}; "
                    f"sampled error {fit.sampled_max_error:.3e}; "
                    f"rank {nodes.t.size}")
                output.append(SharedSigmaWindow(
                    win, branch.E_A, omega_c, omega_idx_c,
                    rows_idx, rows_bounds, rows_phase,
                    band_weight=branch.band_weight))

            for branch, neg, mask_lo, eb_lo in sliver_branches:
                omega_all = np.asarray(branch.omega_abs, np.float64)
                idx_all = np.asarray(branch.omega_idx)
                clusters = _omega_clusters(omega_all, gap_ry)
                if len(clusters) == 1:
                    w_lo = float(np.min(omega_all))
                    w_hi = float(np.max(omega_all))
                    f_max = max(abs(w + e + a)
                                for w in (w_lo, w_hi)
                                for e in eb_lo
                                for a in (a_lo, a_hi))
                    _sliver_damped(
                        branch, neg, mask_lo, eb_lo, branch.omega_abs,
                        branch.omega_idx, f_max, idx, bounds, phase, "",
                        pad=False)
                    continue
                # δ: the same edge-margin protection scale as elsewhere.
                delta = float(edge_factor) * eta
                thresh = delta - eb_lo[0]
                for om_sel, w_lo, w_hi in clusters:
                    omega_c = omega_all[om_sel]
                    omega_idx_c = idx_all[om_sel]
                    if w_lo >= thresh:
                        _sliver_laplace(
                            branch, neg, mask_lo, eb_lo, omega_c,
                            omega_idx_c, w_lo, w_hi, 0.0,
                            " (high-omega cluster)")
                        continue
                    sub_lo = omega_c <= thresh
                    if np.any(~sub_lo):
                        _sliver_laplace(
                            branch, neg, mask_lo, eb_lo,
                            omega_c[~sub_lo], omega_idx_c[~sub_lo],
                            float(np.min(omega_c[~sub_lo])),
                            float(np.max(omega_c[~sub_lo])), 0.0,
                            " (above the crossing corner)")
                    if np.any(sub_lo):
                        w_hi_sub = float(np.max(omega_c[sub_lo]))
                        _sliver_laplace(
                            branch, neg, mask_lo, eb_lo,
                            omega_c[sub_lo], omega_idx_c[sub_lo],
                            float(np.min(omega_c[sub_lo])), w_hi_sub,
                            thresh, " (deep-pole side of the corner)")
                        if a_lo <= thresh:
                            keep = [i for i, row in enumerate(stats)
                                    if row[0] <= thresh]
                            rows_bounds = np.asarray(bounds)[keep].copy()
                            rows_bounds[:, 1] = np.minimum(
                                rows_bounds[:, 1], thresh)
                            f_max = max(
                                w_hi_sub + eb_lo[1] + thresh,
                                abs(eb_lo[0]))
                            _sliver_damped(
                                branch, neg, mask_lo, eb_lo,
                                omega_c[sub_lo], omega_idx_c[sub_lo],
                                f_max, np.asarray(idx)[keep],
                                rows_bounds, np.asarray(phase)[keep],
                                " (crossing corner)", pad=True)

    return output, {"eta_ry": eta, "omega_max_ry": omega_max,
                    "crossing_edge_ry": crossing_edge,
                    "target_error": sector_error,
                    "omega_gap_ry": gap_ry,
                    "n_windows": len(output),
                    "n_tau": int(sum(row.window.n_tau for row in output))}


__all__ = [
    "CROSSING_NODE_FLOOR", "SharedSigmaWindow",
    "build_shared_sigma_windows", "summarize_sigma_poles",
]

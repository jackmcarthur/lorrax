"""MPA Sigma frequency windows, derived directly from fitted pole geometry.

This module reduces each sharded pole field (or a shared-pole store's ragged
census) to the per-branch extrema the box planner (``gw.sigma_box_plan``)
needs, and defines the planner's row type ``SharedSigmaWindow``.  No pole tile,
plan file, or spatial kernel lives here.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gw.efermi import (OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
                       band_in_occupation_window, occupation_weight_floor)
from gw.ppm_windows import _SigmaWindow


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
    #: Green's-function branch owning this row.  Ordered MPA residues use
    #: the same conduction->R+ / valence->R- selection as GN-PPM.
    space: str = ""


_INF = np.inf

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


def sigma_pole_edges(branches, state_edge, excursion):
    """The Σ planner's pole edges, in Ry: the one owner of their formula.

    ``pos``/``neg`` belong to the crossing branch of each ω half
    (``ω≥E_F cond``, ``ω<E_F val``): a pole above ``|ω|max(half) + edge + exc``
    leaves every denominator of that half at least ``edge`` from zero.
    ``near = edge + exc`` is both the pole edge and the ``|ω|`` cut of the
    non-crossing branches (``ω≥E_F val``, ``ω<E_F cond``): there
    ``|d| = |ω| + E + a`` with ``E ≥ -exc`` and ``a > 0``, so ``|ω| ≥ near``
    or ``a > near`` puts ``|d|`` above ``edge``.  The pole selectors are
    ``"all"`` and ``"shallow:<name>"`` / ``"deep:<name>"`` per edge.

    Tempting, and why not: one edge from the global ``ω_max``.  The cover
    grows the ω≥E_F half only, and its edge made the ω<E_F crossing window
    reach 7.9 Ry of poles (Na 8^3 map 0: 409 -> 263 pairs with the half's
    own edge, claim 2821).
    """
    near = float(state_edge) + float(excursion)
    pos, neg = (
        max((float(np.max(b.omega_abs)) for b in branches
             if b.omega_abs.size and bool(b.neg_omega_half) == negative),
            default=0.0)
        for negative in (False, True))
    return {"pos": pos + near, "neg": neg + near, "near": near}


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
    edges = sigma_pole_edges(branches, float(edge_factor) * eta, excursion)
    selectors = {"all": _selector()}
    for name, edge in edges.items():
        selectors[f"shallow:{name}"] = _selector(a_hi=edge)
        selectors[f"deep:{name}"] = _selector()
        selectors[f"deep:{name}"][0] = edge
    return omega_max, eta, edges, selectors


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

    ``occupation_window_threshold`` MUST match the value the branch build and
    the box planner use — they share ``_geometry``, and a mismatch would
    select poles against one support and windows against another.  All come
    from the single deck key in production.
    """
    _omega_max, _eta, _edges, selectors = _geometry(
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


def shared_pole_frequencies(poles2_ry2, counts):
    """Validate replicated metadata and return sorted active Ω per parent.

    Parameters
    ----------
    poles2_ry2 : numpy.ndarray
        Float64 squared frequencies in Ry², shape ``(nparent, Kcap)``.
        Inactive columns carry the store's finite positive sentinel.
    counts : numpy.ndarray
        Int64 active prefix lengths, shape ``(nparent,)``. No matrix or
        factor data is transferred to the host for this census.

    Returns
    -------
    tuple of numpy.ndarray
        Ragged active positive frequencies in Ry. The square root follows
        ``W_c(z) = C (z² - Λ)^-1 C†`` (DESIGN §3.4).
    """
    poles2 = np.asarray(poles2_ry2)
    counts = np.asarray(counts)
    if poles2.ndim != 2 or poles2.dtype != np.dtype(np.float64):
        raise ValueError("shared-pole poles2 must be float64 [parent,Kcap]")
    if counts.shape != poles2.shape[:1] or counts.dtype != np.dtype(np.int64):
        raise ValueError("shared-pole K must be int64 [parent]")
    if poles2.shape[1] > np.iinfo(np.int32).max:
        raise ValueError("shared-pole column indices exceed int32 capacity")
    if np.any(counts < 0) or np.any(counts > poles2.shape[1]):
        raise ValueError("shared-pole K lies outside the stored column extent")
    if not np.all(np.isfinite(poles2)) or np.any(poles2 <= 0):
        raise ValueError("shared-pole frequencies and sentinels must be finite positive")
    result = []
    for row, count in zip(poles2, counts):
        active = row[:int(count)]
        if np.any(active[1:] < active[:-1]):
            raise ValueError("shared-pole active columns must be jointly sorted by Λ")
        result.append(np.sqrt(active))
    return tuple(result)


def shared_pole_intervals(frequencies, pole_indices, bounds):
    """Return parent-prefix intervals for the owner's ``(lower, upper]`` rule.

    ``frequencies`` is the validated ragged census in Ry; ``pole_indices``
    and six-column ``bounds`` are existing planner selector records. The
    returned int32 ``[nparent,2]`` intervals are half-open column ranges.
    Equal poles at a boundary stay together. Damping is identically zero.
    """
    indices = np.asarray(pole_indices)
    bounds = np.asarray(bounds, dtype=np.float64)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("shared-pole parent indices must be an integer vector")
    if bounds.shape != (indices.size, 6) or np.any(np.isnan(bounds)):
        raise ValueError("shared-pole bounds must be [nselected,6] without NaN")
    if np.any(indices < 0) or np.any(indices >= len(frequencies)):
        raise ValueError("shared-pole selector references a missing parent")
    if len(set(map(int, indices))) != indices.size:
        raise ValueError("shared-pole window selects a parent more than once")
    intervals = np.zeros((len(frequencies), 2), dtype=np.int32)
    for parent, b in zip(indices, bounds):
        if not (0 >= b[2] and 0 > b[3] and 0 < b[4] and 0 <= b[5]):
            continue
        if b[1] <= b[0]:
            continue
        intervals[int(parent)] = np.searchsorted(
            frequencies[int(parent)], b[:2], side="right")
    return intervals


def summarize_shared_poles(
    poles2_ry2, counts, branches, *, regularization_width_ry, edge_factor,
    occupation_window_threshold=OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
):
    """Feed real, ragged parent extrema to the existing Σ window planner.

    Metadata shapes/units follow :func:`shared_pole_frequencies`. A parent
    occupies one planner record, irrespective of its rank. Neither fake
    elementwise residue fields nor a frozen pole ceiling enter this census;
    call it again with the current bands, occupations and model each SC map.
    """
    frequencies = shared_pole_frequencies(poles2_ry2, counts)
    _, _, _, selectors = _geometry(
        branches, regularization_width_ry, edge_factor,
        _weight_floor(occupation_window_threshold))
    evidence = [dict() for _ in frequencies]
    indices = np.arange(len(frequencies), dtype=np.int64)
    for name, bounds in selectors.items():
        intervals = shared_pole_intervals(
            frequencies, indices,
            np.broadcast_to(bounds, (len(frequencies), 6)))
        for parent, (lo, hi) in enumerate(intervals):
            values = frequencies[parent]
            evidence[parent][name] = (
                None if lo == hi else
                (float(values[lo]), float(values[hi - 1]), 0.0, 0.0))
    return tuple(enumerate(evidence))


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

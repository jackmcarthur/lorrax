"""Measure-independent denominator-box quadrature for dynamic Sigma(omega).

The public path in this module is deliberately short:

``physical product window -> denominator box -> rule -> executor nodes``.

Pole fields remain distributed.  MPA supplies the bounded per-pole extrema
returned by :func:`gw.mpa.sigma_windows.summarize_sigma_poles`; PPM supplies
the scalar extrema of each exact ``(q, mu, nu)`` pane.  No residue histogram,
sampled lattice, error apportionment, or campaign-wide selection enters the
quadrature.  The box construction, lower-half-plane conjugation, in-run rule
reuse, fit guards, and conversion to executor ``(t, alpha)`` live here once for
both routes.

No rule outlives its process (owner, 2026-09-28: "i really don't want any
cached rules for quadratures at all"): every plan builds its rules cold, the
widest crossing window of the gate decks (1148 nodes) in about 1 s on the
rank's cores (``minimax.uniform_rule._map_rows``).
Within one run a rule is reused only through the in-process request scope
(:func:`_scope_lookup`), so the sector calls of one map share their fits.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import pickle
import time
from collections import OrderedDict
from dataclasses import replace

from ffi import _services

_services.ensure_on_path()

import jax.numpy as jnp
import numpy as np

from common import timing
from common.collectives import (all_gather_processes, gather_to_host,
                                process_count, process_rank)
from common.units import RYD_TO_EV
from gw.minimax_screening import MinimaxNodes
from gw.mpa.sigma_windows import SharedSigmaWindow, sigma_pole_edges
from gw.qp_support import SEMICORE_EPS
from gw.ppm_windows import _SigmaWindow
from minimax import (
    analytic_box_rule,
    analytic_line_box_rule,
    UniformRule,
    boundary_samples,
    rule_roundoff_amplification,
)

#: The box-rule builder: derived nodes and counts, weights from one linear
#: solve (``minimax.analytic_box``), certified on the box boundary; no node is
#: optimized; the widest gate window (1148 nodes) builds in about 1 s.
_BOX_RULE_BUILDER = analytic_box_rule


_FACTOR_GROWTH_CAP = 30.0
_RUNTIME_NOISE_EPSILON = 6.0e-8
#: Absolute budget on a rule's runtime-noise bound (roundoff amplification x
#: _RUNTIME_NOISE_EPSILON, :func:`_accept_rule`), in the certificate's own
#: currency.  Roundoff is set by the executor's arithmetic, not by the
#: quadrature, so its budget does not shrink with eps: production's
#: 0.05 x 1e-4 (the same double), which lets eps below 2e-5 certify.
_RUNTIME_NOISE_BUDGET = 5.0e-6
_SC_POLE_PAD_FRACTION = 0.10
_BOX_SIGN_FRACTION = 0.7
#: Pad toward zero for a sign-definite SC window: 0.5 of its distance escaped by 1.6% on TaAs 8^3
#: map 1 and 0.25 again at map 2 (semimetal valence state 30 -> 15 -> 6 meV from E_F); 0.05 floors it below 1 meV.
_SC_ZERO_SIDE_CAP = 0.05
#: The rule family every digest names: derived rules (bent contour + sector
#: rule, ``minimax.analytic_box``) on outward-snapped boxes, accepted on the
#: term mass in the box's currency.  It enters :func:`_rule_digest`, which
#: orders equal-count candidates when a plan serves a window.
_RULE_FAMILY = "sigma-box-ry-v7"
#: Request scopes held by this process (:func:`_scope_lookup`): the newest
#: few, so a long SC run holds a bounded set.  Sector calls of one map share
#: one scope; a later map with other poles opens its own.
_SCOPE_LIMIT = 4
_SCOPES = OrderedDict()


def _rule_digest(rule, noise_amplification):
    """Name the certificate and its complex128 Ry-inverse nodes."""
    identity = hashlib.sha256(json.dumps(
        [_RULE_FAMILY, list(rule.box), float(rule.eps),
         bool(rule.relative)]).encode())
    for array in (rule.times, rule.weights):
        identity.update(np.asarray(array, dtype="<c16").tobytes())
    identity.update(np.asarray(
        [rule.sup_error, rule.kappa_max, noise_amplification,
         rule.theta_deg, rule.rank], dtype="<f8").tobytes())
    return identity.hexdigest()


#: The self-consistent identity spells its Hamiltonian
#: ``sc_map_{iteration}:{occ_hash}`` (gw/shared_pole_recipe.py), so the map
#: label lives inside a value, not only in the ``iteration_id`` key.
_SC_MAP_LABEL = re.compile(r"^sc_map_\d+:")


def _map_invariant_identity(identity):
    """The identity with its SC map label removed, physics intact.

    Stripping only ``iteration_id`` left the map number in ``hamiltonian``, so
    every SC map opened a fresh request namespace and could never serve a
    containment-compatible rule stored by an earlier map. What remains is the
    physical input the rule depends on; every hit is still re-checked for box
    containment and error currency before use.
    """
    return {key: (_SC_MAP_LABEL.sub("", value) if isinstance(value, str) else value)
            for key, value in identity.items() if key != "iteration_id"}


def _receipt_json(receipt):
    """Serialize the durable quadrature receipt as strict JSON.

    Product windows carry open endpoints (a state tail to ``+inf``, a resonant
    window from ``-inf``); ``json.dumps`` would spell them ``Infinity``, which
    no strict parser reads. An unbounded endpoint is ``null`` and its side is
    its position in the interval pair; any other non-finite value refuses.
    """
    def finite(value):
        if isinstance(value, float) and not np.isfinite(value):
            if np.isnan(value):
                raise ValueError("Sigma quadrature receipt: NaN is not an unbounded edge")
            return None
        if isinstance(value, dict):
            return {key: finite(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite(item) for item in value]
        return value
    return json.dumps(finite(receipt), sort_keys=True, allow_nan=False)


#: The request scope of a route without a pole census (PPM, MPA): one per run.
RUN_SCOPE = "run"


def sigma_rule_scope(identity, poles2, counts, *, eta, eps):
    """The in-process request scope of a shared-pole Sigma call.

    ``identity`` is the model's energy/occupation/recipe provenance;
    ``poles2`` [Nq,K] in Ry**2 and ``counts`` [Nq] are the small host census.
    Map labels are excluded (:func:`_map_invariant_identity`): the sector
    calls of one map (which pass the map's union census) and a later map with
    equal physical inputs share rules, but changed spectra or occupations
    cannot inherit a previous map's plan. Nothing in a scope outlives the
    process. Domain containment and the executor noise/growth gates still run
    on every reuse.
    """
    inputs = _map_invariant_identity(identity)
    digest = hashlib.sha256(json.dumps(
        [_RULE_FAMILY, inputs, float(eta), float(eps)],
        sort_keys=True).encode())
    for row, count in zip(poles2, counts):
        digest.update(np.asarray([count], dtype="<i8").tobytes())
        digest.update(np.asarray(row[:int(count)], dtype="<f8").tobytes())
    return "request_" + digest.hexdigest()


def _resolve_uniform_rule_trace():
    """Return whether the debug-only box trace was requested."""
    return bool(os.environ.get("LORRAX_UNIFORM_RULE_TRACE"))


def _live_states(branch):
    """Return the executor's exact live state support on the host."""
    energy = np.asarray(gather_to_host(branch.E_A), dtype=np.float64)
    base = np.asarray(gather_to_host(branch.base_mask_A), dtype=bool)
    if base.shape != energy.shape:
        base = np.reshape(base, energy.shape)
    live = base & np.isfinite(energy)
    if branch.band_weight is not None:
        weight = np.asarray(
            gather_to_host(branch.band_weight), dtype=np.float64
        ).reshape(energy.shape)
        # The branch builder has already applied the deck's occupation-window
        # threshold.  This final nonzero test mirrors the multiplicative
        # executor and prevents an exactly absent state from widening a box.
        live &= np.isfinite(weight) & (np.abs(weight) > 0.0)
    indices = np.flatnonzero(live.reshape(-1)).astype(np.int32)
    if not indices.size:
        raise ValueError(f"Sigma box branch {branch.tag!r} has no live states")
    return energy, energy.reshape(-1)[indices], indices


def _product_geometry(branches, eta, edge_factor):
    omega_max = max(
        (float(np.max(branch.omega_abs)) for branch in branches
         if branch.omega_abs.size), default=0.0)
    excursion = 0.0
    state_rows = []
    for branch in branches:
        shape, energy, indices = _live_states(branch)
        excursion = max(excursion, -min(float(np.min(energy)), 0.0))
        state_rows.append((shape, energy, indices))
    state_edge = float(edge_factor) * eta
    edges = sigma_pole_edges(branches, state_edge, excursion)
    return state_rows, {
        "omega_max_ry": omega_max,
        "state_edge_ry": state_edge,
        "pole_edges_ry": edges,
        "omega_cut_ry": edges["near"],
        "negative_state_excursion_ry": excursion,
        "edge_factor": float(edge_factor),
    }


def _state_products(branch, state_edge, edges):
    """The sole owner of the product partition of one branch.

    Rows are ``(name, state_lo, state_hi, pole_selector, pole_lo, pole_hi,
    omega_lo, omega_hi)``: half-open ``(lo, hi]`` state and pole intervals and
    ``[lo, hi)`` in ``|ω|``.  A crossing branch (``ω≥E_F cond``, ``ω<E_F val``)
    splits states and poles at its half's edge.  A non-crossing branch keeps
    its bulk (``E > edge``) whole; its resonant-side states split at the
    ``|ω|`` cut ``near``: above it every denominator is at least ``edge`` from
    zero (``omega_tail``, one sign-definite window over all poles), below it
    the poles split at ``near``.

    Tempting, and why not: one resonant window over every ``|ω|`` of a
    non-crossing branch.  Its long side is ``ω_max + Ω_max``, a crossing rule
    over the cover-grown range for a sliver of excursion states near
    ``ω = 0`` (Na 8^3 map 0: 1682 -> 39 pairs, fit 477 -> 2 s, claim 2821).
    """
    crossing = ((branch.space == "cond" and not branch.neg_omega_half)
                or (branch.space == "val" and branch.neg_omega_half))
    inf = np.inf
    if crossing:
        edge = edges["neg" if branch.neg_omega_half else "pos"]
        half = "neg" if branch.neg_omega_half else "pos"
        return (
            ("resonant", -inf, edge, f"shallow:{half}", 0.0, edge, 0.0, inf),
            ("state_tail", edge, inf, f"shallow:{half}", 0.0, edge, 0.0, inf),
            ("pole_tail", -inf, inf, f"deep:{half}", edge, inf, 0.0, inf),
        )
    near = edges["near"]
    return (
        ("bulk", state_edge, inf, "all", 0.0, inf, 0.0, inf),
        ("resonant", -inf, state_edge, "shallow:near", 0.0, near, 0.0, near),
        ("pole_tail", -inf, state_edge, "deep:near", near, inf, 0.0, near),
        ("omega_tail", -inf, state_edge, "all", 0.0, inf, near, inf),
    )


def _pole_rows(summaries, selector):
    indices, stats = [], []
    for pole, evidence in summaries:
        row = evidence[selector]
        if row is not None:
            indices.append(int(pole))
            stats.append(tuple(float(value) for value in row))
    return np.asarray(indices, dtype=np.int32), stats


def _pole_bounds(count, lower, upper):
    bounds = np.asarray(
        (lower, upper, -np.inf, -np.inf, np.inf, np.inf),
        dtype=np.float64)
    return np.broadcast_to(bounds, (int(count), 6)).copy()


def _box(real_lo, real_hi, gamma_lo, gamma_hi, eta):
    """Pad real support by 2%, without changing its sign topology."""
    pad = 0.02 * max(real_hi - real_lo, eta)
    lo = real_lo - pad if real_lo <= 0.0 else max(real_lo - pad,
                                                   _BOX_SIGN_FRACTION * real_lo)
    hi = real_hi + pad if real_hi >= 0.0 else min(real_hi + pad,
                                                   _BOX_SIGN_FRACTION * real_hi)
    return (float(lo), float(hi),
            float(gamma_lo + eta), float(gamma_hi + eta))


def _box_for_window(frequencies, states, pole_stats, pole_sign, eta):
    a_lo = min(row[0] for row in pole_stats)
    a_hi = max(row[1] for row in pole_stats)
    gamma_lo = min(row[2] for row in pole_stats)
    gamma_hi = max(row[3] for row in pole_stats)
    corners = [
        frequency - pole_sign * (state + pole)
        for frequency in (float(np.min(frequencies)),
                          float(np.max(frequencies)))
        for state in (float(np.min(states)), float(np.max(states)))
        for pole in (a_lo, a_hi)
    ]
    raw_lo, raw_hi = float(min(corners)), float(max(corners))
    return (_box(raw_lo, raw_hi, gamma_lo, gamma_hi, eta),
            (raw_lo, raw_hi), (a_lo, a_hi, gamma_lo, gamma_hi))


def make_sigma_box_spec(
    *, name, frequencies, states, pole_stats, pole_sign, eta_ry,
):
    """Construct one route-neutral denominator-box fit specification.

    Parameters
    ----------
    name
        Stable diagnostic identity for the physical product window.
    frequencies
        External frequencies owned by the window, shape ``(nomega,)`` in Ry.
    states
        Exact live intermediate-state energies, shape ``(nstate,)`` in Ry.
    pole_stats
        Per-pole or per-pane ``(real_min, real_max, gamma_min, gamma_max)``
        rows in Ry.  These are scalar extrema, never histogram weights.
    pole_sign
        ``+1`` for conduction denominators and ``-1`` for valence.
    eta_ry
        Positive retarded broadening in Ry.

    Returns
    -------
    dict
        Box, raw support, fit currency, conjugation, and factor references.
        Route-specific selector metadata may be added by the caller.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    states = np.asarray(states, dtype=np.float64).reshape(-1)
    rows = tuple(tuple(float(value) for value in row) for row in pole_stats)
    sign, eta = float(pole_sign), float(eta_ry)
    if not frequencies.size or not np.all(np.isfinite(frequencies)):
        raise ValueError(f"Sigma box window {name!r} has no finite frequencies")
    if not states.size or not np.all(np.isfinite(states)):
        raise ValueError(f"Sigma box window {name!r} has no finite states")
    if not rows or any(len(row) != 4 for row in rows):
        raise ValueError(
            f"Sigma box window {name!r} needs four pole extrema per row")
    if sign not in (-1.0, 1.0):
        raise ValueError("Sigma box pole_sign must be +1 or -1")
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("sigma_quadrature requires eta_ry > 0")
    box, raw_real, pole_extent = _box_for_window(
        frequencies, states, rows, sign, eta)
    kind = ("sign_definite_positive" if box[0] > 0.0 else
            "sign_definite_negative" if box[1] < 0.0 else
            "crossing")
    e_ref_a, e_ref_b = _factor_references(kind, sign, states, rows)
    return {
        "name": str(name), "frequencies": frequencies, "states": states,
        "pole_stats": rows, "pole_sign": sign,
        "raw_real_support": raw_real, "box": box, "kind": kind,
        "pole_extent": pole_extent, "conjugate": sign < 0.0,
        "E_ref_A": e_ref_a, "E_ref_B": e_ref_b,
    }


def _law_node_count(box, eps):
    """Closed-form node count of the rung-0 rule a build on ``box`` takes.

    ``minimax.analytic_box``: ``crossing_nodes`` for a crossing box (the count
    follows the short side over im_lo, the narrow side floored at 4 eta),
    ``sector_degree`` for a sign-definite one (logarithmic in the radial
    ratio). A build that fails rung 0 takes a later rung and more nodes.
    """
    from minimax import crossing_nodes, sector_degree
    box = tuple(float(value) for value in box)
    if box[0] > 0.0 or box[1] < 0.0:
        return int(sector_degree(box, eps)[0])
    return int(crossing_nodes(box, eps)[0].size)


def _may_serve(rule_box, rule_nodes, box, *, ceiling_nodes):
    """THE serving criterion, shared by the scope lookup and the plan.

    A rule serves a request iff its box contains the request's and its node
    count is at most ``ceiling_nodes``, the closed-form count of the
    request's own build. Containment alone let a much larger crossing rule
    serve a small request at several times the cost (a crossing count grows
    with its short side): on MoS2 3x3 SC with the semicore at eta (branch
    feat/qsgw-partition-2026-09-28) the map-0 probe pass's semicore rule
    ([-63.5, +127] eV, 693 nodes) served the omega>=E_F conduction window
    ([-35.5, +17.5] eV, law 203), 1472 instead of 982 pairs per map. A
    foreign rule reaches a request through a sector call or a second plan in
    one request scope; one criterion for both servers.
    """
    return (_box_contains(tuple(float(value) for value in rule_box), box)
            and (ceiling_nodes is None or int(rule_nodes) <= int(ceiling_nodes)))


def _scope_lookup(
    scope, box, eps, relative, *, noise_amplification_cap, ceiling_nodes,
):
    """``(rule, amplification, digest)``: the smallest compatible rule this
    process accepted earlier in ``scope``, or ``None``.

    COMPATIBLE is :func:`_may_serve`: the rule's box contains the request and
    the rule has at most ``ceiling_nodes`` nodes, the closed-form count of
    the request's own build (:func:`_law_node_count`); plus the same error
    currency, noise ceiling and certificate. Equal counts go to the smaller
    digest, a fixed order. ``scope=None`` reuses nothing.
    """
    best = None
    for digest, (rule, amplification) in sorted(_SCOPES.get(scope, {}).items()):
        # A certificate above eps, or one built for a looser noise consumer,
        # is not a rule for this request whatever its node count (Na
        # pole-tail, 2026-09-05).
        if (abs(rule.eps - eps) > 1.0e-12 * eps
                or rule.relative != relative
                or amplification > noise_amplification_cap
                or rule.sup_error > eps):
            continue
        if not _may_serve(rule.box, rule.node_count, box,
                          ceiling_nodes=ceiling_nodes):
            continue
        if best is None or rule.node_count < best[0].node_count:
            best = (rule, amplification, digest)
    return best


def _scope_store(scope, fits):
    """Hold this plan's builds in ``scope`` for the rest of the process.

    Every rank stores the same replicated receipts after the plan's gather,
    so no rank's lookup depends on how far another has got.
    """
    if scope is None:
        return
    entries = _SCOPES.setdefault(scope, {})
    _SCOPES.move_to_end(scope)
    for fit in fits:
        if fit["built"] and _rule_is_certified(fit["rule"], float(fit["rule"].eps)):
            entries.setdefault(fit["rule_digest"],
                               (fit["rule"], fit["roundoff_amplification"]))
    while len(_SCOPES) > _SCOPE_LIMIT:
        _SCOPES.popitem(last=False)


def _rule_is_certified(rule, eps) -> bool:
    """A rule is a rule only if every number in it is finite and sup <= eps.

    A finite ``sup_error`` beside NaN nodes or weights is not a certificate
    (lane QUADCHECK's boundary probe, JID 57930535.148, 2026-09-05): the
    executor would multiply NaN into every Sigma element and the noise gate
    compares NaN with a ceiling, which never refuses.
    """
    times = np.asarray(rule.times)
    weights = np.asarray(rule.weights)
    return bool(
        times.size > 0 and weights.size == times.size
        and np.all(np.isfinite(times)) and np.all(np.isfinite(weights))
        and np.isfinite(float(rule.sup_error))
        and float(rule.sup_error) <= float(eps)
        and np.isfinite(float(rule.kappa_max)))


#: Relative cell of the logarithmic grid every build box is snapped outward to.
_BUILD_GRID_STEP = 1.0e-4


def snap_outward(x, scale, outward):
    """``x`` moved outward (``outward`` = +1 up, -1 down) to the nearest point
    of ``sign(x) * scale * (1 + _BUILD_GRID_STEP)**k``, k integer; zero stays
    zero and a nonzero edge never changes sign (a sign-definite box stays so).

    The one grid for every quantity a rule is built from or a reuse decision
    compares: Sigma build boxes (scale eta), the chi response rule support and
    the SC shared-pole sampling envelope (scale 1). A round-off-perturbed
    input lands on the same grid point unless it straddles a cell edge."""
    if x == 0.0:
        return 0.0
    k = np.log(abs(x) / scale) / np.log1p(_BUILD_GRID_STEP)
    k = np.ceil(k) if outward * np.sign(x) > 0 else np.floor(k)
    return float(np.sign(x) * scale * np.exp(k * np.log1p(_BUILD_GRID_STEP)))


def _build_box(box, eta, *, widen):
    """The box a rule is built on: the request, optionally widened, snapped outward.

    ``widen`` adds 1% of the width to the far edges (|x| > 3 eta) so the
    sector calls of one map reuse by containment. The snap makes the rule a
    function of a grid cell rather than of the exact request: a request
    moved by round-off (extreme shared-pole edges differ 1e-9-4e-8 relative
    between two exact GEMM orders) otherwise lands on a different rule (the
    fitted builder of P2-E moved Fe 4^3 bispinor eqp1 by 0.32 meV this way,
    2026-09-24). On the 1e-4 grid a perturbed request maps to the same build
    box, hence the same rule bit for bit, unless it straddles a cell edge
    (probability ~ perturbation / 1e-4). The cell is kept that fine because
    the count ladder moves in 10% steps: a marginal certification flips when
    its box grows.
    """
    if widen:
        extra = 0.01 * max(box[1] - box[0], eta)
        near = 3.0 * eta
        box = (box[0] - extra if box[0] < -near else box[0],
               box[1] + extra if box[1] > near else box[1],
               box[2], box[3] * 1.01)
    return (snap_outward(box[0], eta, -1), snap_outward(box[1], eta, +1),
            snap_outward(box[2], eta, -1), snap_outward(box[3], eta, +1))


def _factor_references(kind, pole_sign, states, pole_stats):
    if kind == "crossing":
        return float(np.min(states)), 0.0
    table_sign = 1.0 if kind == "sign_definite_positive" else -1.0
    endpoint = np.max if pole_sign * table_sign > 0.0 else np.min
    pole_real = np.asarray(
        [value for row in pole_stats for value in row[:2]], np.float64)
    return float(endpoint(states)), float(endpoint(pole_real))


def _factor_growth(times, pole_sign, states, pole_stats, e_ref_a, e_ref_b):
    """Worst log growth of the executor's two separately factored terms."""
    times_exec = pole_sign * np.asarray(times, np.complex128).reshape(-1)
    green = float(np.max(np.real(
        -1.0j * (states[:, None] - e_ref_a) * times_exec[None, :])))
    pole_corners = np.asarray([
        real - 1.0j * gamma
        for row in pole_stats
        for real in row[:2]
        for gamma in row[2:]
    ], dtype=np.complex128)
    screened = float(np.max(np.real(
        -1.0j * (pole_corners[:, None] - e_ref_b)
        * times_exec[None, :])))
    return green, screened


def _noise_amplification_cap():
    """Largest roundoff amplification the executor's noise budget admits."""
    return _RUNTIME_NOISE_BUDGET / _RUNTIME_NOISE_EPSILON


def _fit_rule(spec, eps, scope, eta, *, build_widen=True):
    """Reuse or build one window's rule and accept it; never store it.

    The scope lookup reads only rules accepted before this plan: the plan's
    own builds are stored after every rank has looked up (see
    :func:`fit_sigma_box_spec_groups`), so no window's choice depends on how
    far another rank has got.
    """
    requested_box = spec["box"]
    # This is exactly the builder's default currency predicate.  It is used
    # here only to match reused rules; a build still leaves the choice to
    # _BOX_RULE_BUILDER(relative=None).
    relative = requested_box[0] > 0.0 or requested_box[1] < 0.0
    noise_amplification_cap = _noise_amplification_cap()
    analytic_line = (bool(spec.get("analytic_line")) and not relative
                     and requested_box[2] == requested_box[3]
                     and spec["pole_extent"][2:] == (0.0, 0.0))
    built, amplification = False, None
    if analytic_line:
        # PPM's real poles make Im(d)=eta exactly.  Ask the analytic service
        # for that line; a reused rectangle rule cannot silently preempt it.
        rule = analytic_line_box_rule(requested_box, eps)
        rule_source = "analytic-line"
    else:
        build_box = _build_box(requested_box, eta, widen=build_widen)
        ceiling_nodes = _law_node_count(build_box, eps)
        reused = _scope_lookup(
            scope, requested_box, eps, relative,
            noise_amplification_cap=noise_amplification_cap,
            ceiling_nodes=ceiling_nodes)
        if reused is not None:
            rule, amplification, digest = reused
            rule = replace(rule, seconds=0.0)
            rule_source = f"scope:{digest[:16]}"
        else:
            # The builder's ladder steps on the executor's own noise gate:
            # the term mass rho*sum|w e^{itd}| in the box's currency (rho =
            # |d| or eta), the quantity _accept_rule bounds below.
            rule = _BOX_RULE_BUILDER(build_box, eps,
                                     mass_cap=noise_amplification_cap)
            built = True
            rule_source = "built"
        # There is no retry.  The builder takes no clock and no pass count,
        # so a second call with the same inputs returns the same rule; the
        # old 5x-budget retry existed only because the first attempt could
        # have been cut short by a deadline, and there is no deadline to
        # lengthen.  A refusal here is now a statement about the box.
    fit = _accept_rule(spec, rule, eps, rule_source=rule_source,
                       noise_amplification=amplification)
    fit.update(built=built, analytic_line=analytic_line,
               serve_ceiling_nodes=None if analytic_line else ceiling_nodes)
    return fit


def _accept_rule(spec, rule, eps, *, rule_source, noise_amplification=None):
    """Accept one rule for one window, or refuse; return its executor receipt.

    ``noise_amplification`` is the rule's own term mass when an earlier
    acceptance in this process already measured it (a function of the rule
    and ``eps`` only); ``None`` measures it here.
    """
    noise_budget = _RUNTIME_NOISE_BUDGET
    # ONE ACCEPTANCE ON EVERY PATH.  One-shot, fixed-SC initialization and
    # its rebuilds all require the certified sup error at or below eps; the
    # fixed-SC bypass (enforce_sup_error=False, 2026-09-03) let Na retain a
    # conduction pole-tail rule at 400 x eps in every self-consistent arm.
    if not _rule_is_certified(rule, eps):
        raise RuntimeError(
            f"Sigma box window {spec['name']!r} refused: rule sup error "
            f"{float(rule.sup_error):.6g} exceeds eps={eps:.6g} or the rule "
            f"is not finite ({int(np.asarray(rule.times).size)} nodes on box "
            f"{tuple(round(float(v), 6) for v in rule.box)}, kind "
            f"{spec.get('kind', '?')}, rule={rule_source}"
            f"). Remedy: a sign-preserving or split product window (the SC "
            f"pad now keeps sign-definite supports sign-definite), or a "
            f"certified crossing rule; do not loosen sigma_quadrature_eps to "
            f"admit this rule.")
    # Runtime perturbations must be bounded in the SAME currency as the
    # approximation.  ``kappa = sum|term|/|Q|`` is already relative for a
    # sign-definite box, but it overstates a crossing box's peak-relative
    # error by ~|d|/eta at its far edge.  Measure rho*sum|term| directly.
    # the noise mass has a subharmonic logarithm, so its box maximum lies on
    # the boundary: sample the edges at the rule's own horizon
    if noise_amplification is None:
        noise_cloud = boundary_samples(
            rule.box, rule.theta_deg, float(np.max(np.abs(rule.times))), eps)
        noise_rho = (np.abs(noise_cloud) if rule.relative
                     else float(np.min(noise_cloud.imag)))
        noise_amplification = rule_roundoff_amplification(
            rule.times, rule.weights, noise_cloud, noise_rho)
    noise_bound = noise_amplification * _RUNTIME_NOISE_EPSILON
    if not np.isfinite(noise_bound) or noise_bound > noise_budget:
        raise RuntimeError(
            f"Sigma box window {spec['name']!r} refused: runtime-noise "
            f"bound {noise_bound:.6g} exceeds {noise_budget:.6g}")

    times = np.asarray(rule.times, np.complex128)
    weights = np.asarray(rule.weights, np.complex128)
    if spec["conjugate"]:
        # 1/conj(d) = conj(1/d): one upper-half-plane build serves the
        # lower-half-plane causal branch exactly.
        times, weights = -np.conj(times), np.conj(weights)
    growth = _factor_growth(
        times, spec["pole_sign"], spec["states"], spec["pole_stats"],
        spec["E_ref_A"], spec["E_ref_B"])
    if max(growth) > _FACTOR_GROWTH_CAP:
        raise RuntimeError(
            f"Sigma box window {spec['name']!r} refused: factored log "
            f"growth {max(growth):.6g} exceeds {_FACTOR_GROWTH_CAP:g}")
    node_digest = hashlib.sha256(
        np.ascontiguousarray(times).view(np.uint8).tobytes()
        + np.ascontiguousarray(weights).view(np.uint8).tobytes()
    ).hexdigest()[:16]
    return {
        "times": times, "weights": weights,
        "node_count": int(times.size), "rule_box": tuple(rule.box),
        "relative": bool(rule.relative), "sup_error": float(rule.sup_error),
        "kappa_max": float(rule.kappa_max), "theta_deg": float(rule.theta_deg),
        "rank": int(rule.rank), "seconds": float(rule.seconds),
        "rule_source": rule_source, "factor_growth": growth,
        "noise_bound": noise_bound, "noise_budget": noise_budget,
        "roundoff_amplification": noise_amplification,
        "node_digest": node_digest,
        "rule": rule, "rule_digest": _rule_digest(rule, noise_amplification),
        "one_line": (f"analytic line: {rule.node_count} nodes, "
                     f"sup {rule.sup_error:.2e} (eps {eps:g})"
                     if rule_source == "analytic-line" else rule.one_line()),
    }


def _serve_from_plan(specs, fits, eps):
    """Give each window the smallest compatible rule of this whole plan.

    Resolution runs after every build, on the replicated receipts, in a fixed
    order: candidates are the window's own rule (a scope reuse or its build)
    and every rule this plan built, ranked by (node count, certificate
    digest), and a candidate must pass :func:`_may_serve`, the criterion the
    scope lookup applies. The result does not depend on rank timing:
    before this, whether a window saw another window's fresh rule depended on
    how far the other rank had got (Si shared-pole ``cond:pole_tail`` took
    the 9-node own rule or the 7-node ``cond:bulk`` one, eqp1 0.80 ueV
    apart).
    """
    fresh = sorted((fit for fit in fits if fit["built"]),
                   key=lambda fit: (fit["node_count"], fit["rule_digest"]))
    cap = _noise_amplification_cap()
    served = []
    for spec, own in zip(specs, fits):
        chosen = own
        spec_eps = _spec_eps(spec, eps)
        if not own["analytic_line"]:
            box = spec["box"]
            relative = box[0] > 0.0 or box[1] < 0.0
            own_key = (own["node_count"], own["rule_digest"])
            for other in fresh:
                if (other["node_count"], other["rule_digest"]) >= own_key:
                    break
                rule = other["rule"]
                if (abs(rule.eps - spec_eps) > 1.0e-12 * spec_eps
                        or bool(rule.relative) != relative
                        or other["roundoff_amplification"] > cap
                        or not _may_serve(rule.box, other["node_count"], box,
                                          ceiling_nodes=own.get("serve_ceiling_nodes"))):
                    continue
                try:
                    chosen = _accept_rule(
                        spec, rule, spec_eps,
                        rule_source=f"plan:{other['rule_digest'][:16]}",
                        noise_amplification=other["roundoff_amplification"])
                except RuntimeError:
                    continue
                chosen.update(built=False, analytic_line=False)
                break
        served.append(chosen)
    return served


def _spec_eta(spec, eta):
    """A window's own broadening (Ry): a coarse window carries ``eta_ry``
    (:func:`plan_sigma_windows` ``omega_eta_ry``), every other one the plan's."""
    return float(spec.get("eta_ry", eta))


def _spec_eps(spec, eps):
    """A window's own certificate tolerance: a coarse window carries ``eps``
    (``qp_support.SEMICORE_EPS``, never tighter than the plan's), every other
    one the plan's."""
    return float(spec.get("eps", eps))


def _fit_cost(spec, eta):
    """Predicted builder cost, only to balance ranks: a crossing rule's weight
    solve grows with its node count, which follows the box width in units of
    eta; a sign-definite rule is a few tens of nodes."""
    if spec["kind"] != "crossing":
        return 1.0
    return 1.0 + (float(spec["box"][1]) - float(spec["box"][0])) / eta


def _rank_assignment(costs, world):
    """Longest predicted fit first, each to the least-loaded rank.

    Equal costs reduce to round-robin. The assignment is a function of the
    specs, so every rank computes the same one, and a rule does not depend on
    the rank that fits it."""
    load, owned = [0.0] * world, [[] for _ in range(world)]
    for index in sorted(range(len(costs)), key=lambda i: (-costs[i], i)):
        target = min(range(world), key=lambda r: (load[r], r))
        owned[target].append(index)
        load[target] += costs[index]
    return owned


def _parallel_fits(specs, worker, costs):
    """Fit independent windows once across ranks and replicate small rules.

    Each window is built whole on one rank (:func:`_rank_assignment`) and
    every rank uses the gathered bytes, so no rank's plan depends on its own
    BLAS; the bytes are the same under any binding of 16 or more CPUs per
    rank, see
    ``minimax.uniform_rule._BLAS_THREADS``. ``worker(index)``.
    """
    rank, world = int(process_rank()), int(process_count())
    local = []
    for index in _rank_assignment(list(costs), world)[rank]:
        started = time.perf_counter()
        try:
            value = worker(index)
            error = None
        except Exception as exc:  # refusals cross ranks as data, then raise
            value = None
            error = f"{type(exc).__name__}: {exc}"
        local.append({
            "index": index, "source_rank": rank, "value": value,
            "error": error, "wall_seconds": time.perf_counter() - started,
        })
    if world == 1:
        shards = [local]
    else:
        payload = np.frombuffer(
            pickle.dumps(local, protocol=pickle.HIGHEST_PROTOCOL),
            dtype=np.uint8)
        lengths = np.asarray(all_gather_processes(
            np.asarray(payload.size, np.int32)), dtype=np.int64).reshape(-1)
        # A power-of-two carrier: the gather's shape is part of its compile
        # key, and the pickled receipts' size follows the run's paths and
        # windows, so an exact width gave every run its own executable.
        width = 1 << max(0, int(np.max(lengths)) - 1).bit_length()
        padded = np.zeros(width, np.uint8)
        padded[:payload.size] = payload
        gathered = np.asarray(all_gather_processes(padded), np.uint8)
        shards = [pickle.loads(np.ascontiguousarray(
            gathered[source, :int(length)]).tobytes())
                  for source, length in enumerate(lengths)]
    rows = sorted((row for shard in shards for row in shard),
                  key=lambda row: row["index"])
    if [row["index"] for row in rows] != list(range(len(specs))):
        raise RuntimeError("Sigma box planner did not gather every window")
    refusal = next((row for row in rows if row["error"] is not None), None)
    if refusal is not None:
        raise RuntimeError(refusal["error"])
    return [row["value"] for row in rows], rows


def fit_sigma_box_specs(specs, eta_ry, *, eps, scope, build_widen=True):
    """One plan: :func:`fit_sigma_box_spec_groups` with a single group."""
    (fits, fit_rows), = fit_sigma_box_spec_groups(
        [(specs, build_widen)], eta_ry, eps=eps, scope=scope)
    return fits, fit_rows


def fit_sigma_box_spec_groups(groups, eta_ry, *, eps, scope):
    """Fit independent route-neutral box specifications across processes.

    The input rows must come from :func:`make_sigma_box_spec`.  This function
    owns the in-run reuse and the build, rule acceptance, lower-half-plane
    conjugation, runtime-noise guard, and factored-growth guard.  It returns
    only small replicated rule receipts; route-specific physical selectors
    stay with the caller.  Every window is looked up in ``scope`` as it was
    before the plan (:func:`_scope_lookup`), every other window is built, the
    builds join the scope, and each window is then served the smallest
    compatible rule of the plan in a fixed order (:func:`_serve_from_plan`),
    so the result does not depend on rank timing.  ``scope=None`` reuses
    nothing across plans.

    ``groups`` is a list of ``(specs, build_widen)``; each group is one plan,
    looked up in the scope as it was before this call and served only from
    its own builds, but all groups share one balanced parallel pass (the SC
    map-0 one-shot and padded sets, P2-E 2026-09-24). ``build_widen`` widens
    the far edges of a build box by 1% (:func:`_build_box`), so the sector
    calls of one map reuse by containment. Returns one ``(fits, fit_rows)``
    per group.
    """
    rows, widen, bounds = [], [], []
    for specs, group_widen in groups:
        start = len(rows)
        rows.extend(specs)
        widen.extend([bool(group_widen)] * (len(rows) - start))
        bounds.append((start, len(rows)))
    eta, tolerance = float(eta_ry), float(eps)
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("sigma_quadrature requires eta_ry > 0")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("sigma_quadrature_eps must lie in (0, 1)")
    fits, fit_rows = _parallel_fits(
        rows, lambda index: _fit_rule(
            rows[index], _spec_eps(rows[index], tolerance), scope,
            _spec_eta(rows[index], eta), build_widen=widen[index]),
        [_fit_cost(spec, _spec_eta(spec, eta)) for spec in rows])
    # Every rank has looked up by now (the gather above) and holds the same
    # replicated receipts, so storing the plan's builds on every rank cannot
    # change any choice made in it, and the next plan sees the same scope on
    # every rank.
    _scope_store(scope, fits)
    return [(_serve_from_plan(rows[lo:hi], fits[lo:hi], tolerance),
             fit_rows[lo:hi]) for lo, hi in bounds]


def _box_contains(outer, inner):
    """Return whether one certified denominator box contains another."""
    return (outer[0] <= inner[0] and outer[1] >= inner[1]
            and outer[2] <= inner[2] and outer[3] >= inner[3])


def _box_escape_reasons(outer, inner):
    """Describe every edge by which ``inner`` escapes ``outer``."""
    labels = ("real_lo", "real_hi", "imag_lo", "imag_hi")
    escaped = (
        inner[0] < outer[0], inner[1] > outer[1],
        inner[2] < outer[2], inner[3] > outer[3],
    )
    return [
        f"{label}: current={inner[index]:.12g} Ry, "
        f"fixed={outer[index]:.12g} Ry"
        for index, (label, is_outside) in enumerate(zip(labels, escaped))
        if is_outside
    ]


#: The far pole edge of a window whose selector has no upper bound (deep and
#: bulk windows) is certified to this multiple of its current value.  The
#: highest shared-pole mode moves 10-30% per SC map (Fe 4^3 charge map 2:
#: val:bulk 24.2 -> 28.6 Ry refit 6/12 windows under the old 10%), and a
#: sign-definite relative rule pays about one node for a doubled far edge.
_SC_FAR_POLE_FACTOR = 2.0


def _sc_padded_box_spec(spec, eta, *, occupation_reach_ry=None):
    """Return the held SC certificate box for one product window.

    The SC window plan (``scissor.SC_WINDOW_PAD_EV``, owner 2026-09-25 and
    2026-09-28) certifies each window over the grid it is planned on:

    * its live states. The outer edge (farthest from mu in the branch's own
      coordinate: E - mu on a conduction branch, mu - E on a valence one) is
      padded by ``scissor.sc_window_pad_ev`` = max(2 eV, 10% of |E - mu|).
      The inner edge of a crossing window is padded by
      ``scissor.SC_WINDOW_INNER_PAD_ETA`` eta and never past
      ``-occupation_reach_ry`` (a metal's floor reach,
      ``efermi.occupation_floor_reach_ry``; None on an insulator); a
      sign-definite window keeps the outer pad on both edges. Both edges stop
      at the window's own selector interval;
    * its poles: near edges and widths by ten percent, the far edge of an
      unbounded selector by :data:`_SC_FAR_POLE_FACTOR`.

    A map whose current box stays inside keeps the rule; a map that leaves it
    is an escape, and only that window is rebuilt, by this same rule around
    its current states.

    Why the inner edge of a crossing window is deliberately tight: it sets the
    short side |omega|max + x - Omega_min, which sets the node count, and a
    2 eV pad there cost 18-26 nodes per crossing window (claims 2877, 2936).
    The trade: that slack also absorbed motion the tight edge now rebuilds.
    Any inward motion of the inner state (MoS2 3x3 map 2: the gap edge came
    back 1.5 eV after its map-1 overshoot), any grid extension on the
    crossing half (Fe 4^3 map 2: 26.0 -> 28.25 eV) and any drop of the near
    pole edge past its 10% pad (Fe 4^3 map 1: 0.28 -> 0.19 eV) is an escape,
    rebuilt and logged (Fe 4^3 charge SC-3: 12 windows rebuilt against 9).
    Only a metal's clip is a bound: no branch state passes -X.

    Tempting, and why not: a flat outer pad. QP corrections stretch the
    spectrum by about 10%, so a flat 1 eV pad refit Na 8^3's two crossing
    windows at map 1 (553 s; top state +96 -> +101 eV). The outer edge sets
    only the long side, which costs almost nothing past three short sides
    (claim 2875).
    """
    a_lo, a_hi, gamma_lo, gamma_hi = spec["pole_extent"]
    frac = _SC_POLE_PAD_FRACTION
    # Tempting, and why not: clamping the padded poles to the window's
    # selector bounds. Those bounds are this map's grid edges and move with
    # the grid (MoS2 3x3 SC-3 W10: the top grows 12.5 -> 13.0 eV at map 1, and
    # a clamped cond:resonant escaped at maps 1 and 2, 154 -> 115 -> 142 nodes;
    # QAUDIT 2026-09-27). The sign gap below keeps the zero side off the pads.
    open_above = not np.isfinite(spec.get("pole_bounds", (0.0, 0.0))[1])
    padded_poles = [(
        a_lo - frac * abs(a_lo),
        (_SC_FAR_POLE_FACTOR * a_hi if open_above and a_hi > 0.0
         else a_hi + frac * abs(a_hi)),
        max(0.0, gamma_lo - frac * abs(gamma_lo)),
        gamma_hi + frac * abs(gamma_hi),
    )]
    # A treatment ceiling is a fixed-domain contract, not a live pole. Keep
    # its declared support separate from the current pole statistics and
    # factor references, and use it only to size the frozen certificate.
    if "sc_support_pole_extent" in spec:
        padded_poles.append(tuple(spec["sc_support_pole_extent"]))
    from gw.scissor import SC_WINDOW_INNER_PAD_ETA, sc_window_pad_ev
    states = np.asarray(spec["states"], dtype=np.float64)
    low, high = float(np.min(states)), float(np.max(states))
    pad_hi = float(sc_window_pad_ev(high * RYD_TO_EV)) / RYD_TO_EV
    if spec["kind"] == "crossing":
        inner = low - SC_WINDOW_INNER_PAD_ETA * float(eta)
        if occupation_reach_ry is not None:
            # Never past the reach, and never inside a live state.
            inner = min(low, max(inner, -float(occupation_reach_ry)))
        pad_lo = low - inner
    else:
        pad_lo = float(sc_window_pad_ev(low * RYD_TO_EV)) / RYD_TO_EV
    # Membership is re-selected on every map: a state past the window's own
    # selector bound belongs to the neighbouring window, whose certificate
    # covers it. Padding across the bound only drags a sign-definite edge
    # toward zero until the zero-side cap stops it (TaAs 4^3 metal SC,
    # 2026-09-24: val:bulk at 0.0013 Ry against a 0.0276 Ry selector edge,
    # a 14.6 Ry-tall relative box that certified with 0.04% margin and was
    # refused on refit).
    state_lo, state_hi = spec.get("state_interval", (-np.inf, np.inf))
    padded_states = np.asarray(
        [max(low - pad_lo, state_lo), min(high + pad_hi, state_hi)])
    frequencies = np.asarray(spec["frequencies"], dtype=np.float64)
    pole_box, _, _ = _box_for_window(
        frequencies, padded_states, padded_poles, spec["pole_sign"], eta)
    box = [
        min(spec["box"][0], pole_box[0]),
        max(spec["box"][1], pole_box[1]),
        min(spec["box"][2], pole_box[2]),
        max(spec["box"][3], pole_box[3]),
    ]
    # A SIGN-DEFINITE SUPPORT STAYS SIGN-DEFINITE.  The pads above can
    # push the zero-side edge of a strictly negative (or positive) support
    # across zero, which turns an easy relative rule into a crossing rule
    # the builder cannot certify: the Na conduction pole-tail window was
    # retained at sup=0.04 against eps=1e-4 with 906 nodes, while its actual
    # support has a 24-node rule at eps (lane QUADCHECK, 2026-09-05).  The
    # pad toward zero stops the zero-side edge at ``_SC_ZERO_SIDE_CAP`` of
    # its distance to zero; a support that really crosses later is a box
    # escape and rebuilds.
    if spec["kind"] == "sign_definite_negative":
        box[1] = min(box[1], _SC_ZERO_SIDE_CAP * spec["box"][1])
    elif spec["kind"] == "sign_definite_positive":
        box[0] = max(box[0], _SC_ZERO_SIDE_CAP * spec["box"][0])
    # Membership can change without appreciable state motion: a state just
    # outside a tail at map 0 can enter it at map 1. Where the selectors
    # guarantee a sign gap, every member on every map has |d| >= gap, so the
    # zero-side edge is the gap itself: covered, and never dragged closer by
    # the pads (the Fe 4^3 SC cond:pole_tail pads reached -0.11 eta against a
    # 1.5 eta selector gap, an ill-conditioned box the rule refused, 2026-09-27).
    if "sc_selector_gap_ry" in spec:
        gap = float(spec["sc_selector_gap_ry"])
        if spec["kind"] == "sign_definite_negative":
            box[1] = max(-gap, spec["box"][1])
        elif spec["kind"] == "sign_definite_positive":
            box[0] = min(gap, spec["box"][0])
    padded = dict(spec)
    padded["box"] = tuple(float(value) for value in box)
    padded["kind"] = (
        "sign_definite_positive" if box[0] > 0.0 else
        "sign_definite_negative" if box[1] < 0.0 else "crossing")
    padded["sc_unpadded_box"] = tuple(spec["box"])
    padded["sc_state_pad_ev"] = (pad_lo * RYD_TO_EV, pad_hi * RYD_TO_EV)
    padded["sc_state_extent_ry"] = (float(np.min(states)), float(np.max(states)))
    padded["sc_certified_states_ry"] = tuple(float(v) for v in padded_states)
    padded["sc_certified_poles_ry"] = tuple(float(v) for v in padded_poles[0])
    padded["sc_certified_omega_ry"] = (float(frequencies.min()), float(frequencies.max()))
    padded["sc_pole_pad_fraction"] = _SC_POLE_PAD_FRACTION
    if not _box_contains(padded["box"], spec["box"]):
        raise RuntimeError(
            f"SC fixed-rule padding failed to contain {spec['name']!r}")
    return padded


class _RuleValidityFailure(RuntimeError):
    """The frozen rule is numerically invalid on the current map."""


def _fixed_fit_for_spec(entry, spec):
    """Reuse one immutable rule and recheck current factor growth."""
    fit = dict(entry["fit"])
    growth = _factor_growth(
        fit["times"], spec["pole_sign"], spec["states"],
        spec["pole_stats"], spec["E_ref_A"], spec["E_ref_B"])
    if max(growth) > _FACTOR_GROWTH_CAP:
        raise _RuleValidityFailure(
            f"GATE sc_fixed_rule_validity: Sigma box window {spec['name']!r} "
            f"refused while reusing its fixed SC rule: factored log growth "
            f"{max(growth):.6g} exceeds {_FACTOR_GROWTH_CAP:g}")
    fit["factor_growth"] = growth
    fit["rule_source"] = "hit:sc-fixed"
    fit["seconds"] = 0.0
    return fit


def _escape_attribution(entry, spec):
    """Name what left a held window's certificate: a state, the poles or the
    grid, in eV.  The receipt of the plan that certified it
    (:func:`_sc_padded_box_spec`) holds the certified state, pole and
    frequency intervals; the box edges are the fallback (a sign-definite
    window's zero-side cap)."""
    certified = entry.get("certified")
    if not certified:
        return ""
    parts = []
    states = np.asarray(spec["states"], dtype=np.float64)
    lo, hi = certified["states_ry"]
    outside = np.flatnonzero((states < lo) | (states > hi))
    if outside.size:
        worst = outside[np.argmax(np.maximum(lo - states[outside],
                                              states[outside] - hi))]
        k, n = np.unravel_index(int(spec["state_indices"][worst]),
                                spec["state_shape"])
        # A valence branch carries mu - E (ppm_windows: H_val = -energy).
        branch = spec.get("branch")
        sign = -1.0 if getattr(branch, "space", "cond") == "val" else 1.0
        edges = sorted((sign * lo * RYD_TO_EV, sign * hi * RYD_TO_EV))
        parts.append(
            f"state k={int(k)} band={int(n) + 1} (Sigma band carrier) at "
            f"E-mu={sign * states[worst] * RYD_TO_EV:+.4f} eV left the certified "
            f"[{edges[0]:+.4f}, {edges[1]:+.4f}] eV "
            f"({outside.size} state(s) outside)")
    a_lo, a_hi = spec["pole_extent"][:2]
    p_lo, p_hi = certified["poles_ry"][:2]
    if a_lo < p_lo or a_hi > p_hi:
        parts.append(
            f"poles [{a_lo * RYD_TO_EV:.4f}, {a_hi * RYD_TO_EV:.4f}] eV left the "
            f"certified [{p_lo * RYD_TO_EV:.4f}, {p_hi * RYD_TO_EV:.4f}] eV")
    frequencies = np.asarray(spec["frequencies"], dtype=np.float64)
    w_lo, w_hi = certified["omega_ry"]
    if frequencies.min() < w_lo or frequencies.max() > w_hi:
        parts.append(
            f"grid [{frequencies.min() * RYD_TO_EV:+.3f}, "
            f"{frequencies.max() * RYD_TO_EV:+.3f}] eV left the certified "
            f"[{w_lo * RYD_TO_EV:+.3f}, {w_hi * RYD_TO_EV:+.3f}] eV")
    return "; ".join(parts) if parts else "sign-definite zero-side edge"


def _certified_entry(fit, padded_spec, spec, **extra):
    """One held window: its rule, certificate box and certified intervals."""
    return dict({
        "fit": fit,
        "padded_box": tuple(padded_spec["box"]),
        "initial_box": tuple(spec["box"]),
        "pad_ev": tuple(float(v) for v in padded_spec["sc_state_pad_ev"]),
        "certified": {
            "states_ry": padded_spec["sc_certified_states_ry"],
            "poles_ry": padded_spec["sc_certified_poles_ry"],
            "omega_ry": padded_spec["sc_certified_omega_ry"],
        },
    }, **extra)


def _fit_fixed_sc_rules(
    specs, eta, *, eps, scope, session, material_class=None,
    occupation_reach_ry=None,
):
    """The SC window plan's rules: plan once at map 0, hold, rebuild on an escape.

    Owner 2026-09-25 and 2026-09-28 (``scissor.SC_WINDOW_PAD_EV``): plan the
    windows once, with a pad that most runs never leave, then hold them.

    * Map 0 is served by the ordinary one-shot rules, so SC map 0 equals the
      one-shot G0W0 bit for bit, and in the same balanced pass certifies the
      plan: every window padded on its outer state edge by max(2 eV, 10%
      |E - mu|), and a crossing window on its inner edge by 2 eta, clipped at
      a metal's occupation reach, over the map-0 grid
      (:func:`_sc_padded_box_spec`).
    * Every later map holds. A window whose current box leaves its rule's box
      (a state, the pole extent or the grid edge crossed;
      :func:`_escape_attribution` names it) is an escape: it is rebuilt by
      the same rule around its current states, logged by name, and held
      again. ``escape_maps`` counts the maps with an escape and
      ``rebuild_count`` the windows rebuilt, over the run.
    * A metal<->insulator flip re-initializes the plan; a rule-validity
      failure during reuse (factored-log growth above the cap) rebuilds that
      window the same way.

    The window executables carry the session's largest node count
    (``session["tau_capacity"]``, never lowered), so a refit recompiles them
    only if it raises it.
    """
    from gw.scissor import (SC_WINDOW_INNER_PAD_ETA, SC_WINDOW_PAD_EV,
                            SC_WINDOW_PAD_FRACTION)

    rows = list(specs)
    iteration = int(session.get("call_count", 0)) + 1
    session["call_count"] = iteration
    named_class = None if material_class is None else str(material_class)
    if named_class is not None:
        previous_class = session.get("material_class")
        if previous_class is None:
            session["material_class"] = named_class
        elif str(previous_class) != named_class:
            session["material_class"] = named_class
            session.pop("rules", None)
            session["class_flip"] = f"{previous_class}->{named_class}"
    if "rules" in session and (
            float(session["eta_ry"]) != float(eta)
            or float(session["eps"]) != float(eps)):
        raise ValueError(
            "SC fixed quadrature session changed currency: "
            f"eta {session['eta_ry']!r}->{eta!r}, "
            f"eps {session['eps']!r}->{eps!r}")

    def receipt(mode, fits, *, event, rebuilt=(), reasons=(), escaped=0,
                initialized=False):
        session["tau_capacity"] = max(int(session.get("tau_capacity", 0)), max(
            (int(fit["node_count"]) for fit in fits), default=0))
        return {
            "iteration": iteration, "mode": mode, "initialized": initialized,
            "plan_event": event,
            "rebuilt": tuple(rebuilt), "recompute_reasons": tuple(sorted(reasons)),
            "escaped": int(escaped),
            "rebuild_count_total": int(session.get("rebuild_count", 0)),
            "escape_maps_total": int(session.get("escape_maps", 0)),
            "material_class": session.get("material_class"),
            "class_flip": session.pop("class_flip", None),
            "pad_ev": float(SC_WINDOW_PAD_EV),
            "pad_fraction": float(SC_WINDOW_PAD_FRACTION),
            "inner_pad_eta": float(SC_WINDOW_INNER_PAD_ETA),
            "occupation_reach_ry": occupation_reach_ry,
            "tau_capacity": int(session["tau_capacity"]),
        }

    if "rules" not in session:
        # MAP 0 (or the map of a class flip): served by the one-shot rules, so
        # SC map 0 is the one-shot G0W0 bit for bit; in the same balanced
        # pass the first plan's held set is certified at the first pad.
        session["eta_ry"] = float(eta)
        session["eps"] = float(eps)
        padded = [_sc_padded_box_spec(spec, _spec_eta(spec, eta),
                                      occupation_reach_ry=occupation_reach_ry)
                  for spec in rows]
        (served, fit_rows), (fits, padded_rows) = fit_sigma_box_spec_groups(
            [(rows, True), (padded, False)], eta, eps=eps, scope=scope)
        session["rules"] = {
            spec["name"]: _certified_entry(
                dict(fit, rule_source=f"init:{fit['rule_source']}"), padded_spec, spec)
            for spec, padded_spec, fit in zip(rows, padded, fits)}
        session["initial_window_tau_pairs"] = int(sum(
            fit["node_count"] for fit in fits))
        # The executables serve the one-shot rules on this map and the
        # held ones afterwards: size the node capacity for both.
        return served, list(fit_rows) + list(padded_rows), receipt(
            "one-shot", list(served) + list(fits), event="plan",
            initialized=True)

    rules = session["rules"]
    # From map 1 the rules are held: the plan's margin absorbs the map-to-map
    # motion (Na 8^3: the top state's +5 eV sits inside its 9.6 eV pad), and a
    # window is rebuilt only when its current box leaves its rule. Re-padding
    # every window at map 1 refit 8 of Na's 10 windows (the 10% far-state pad
    # moves with the state), which is the fit the plan exists to avoid.
    reasons_by_name = {}
    for spec in rows:
        entry = rules.get(spec["name"])
        if entry is None or "fit" not in entry:
            # A resumed SC process holds the window names and boxes, never a
            # rule (``gw.sc_iteration._held_session``): it re-certifies here.
            reasons_by_name[spec["name"]] = (
                "escape: absent when the rules froze" if entry is None else
                "escape: resumed process (rules are never stored)")
            continue
        reasons = _box_escape_reasons(entry["fit"]["rule_box"], spec["box"])
        if bool(entry["fit"]["relative"]) != (spec["kind"] != "crossing"):
            reasons.append("absolute/relative error currency changed")
        if reasons:
            reasons_by_name[spec["name"]] = (
                f"escape: {_escape_attribution(entry, spec)} ({'; '.join(reasons)})")
    fit_rows = []
    refit = [spec for spec in rows if spec["name"] in reasons_by_name]
    if refit:
        session["escape_maps"] = int(session.get("escape_maps", 0)) + 1
        padded = [_sc_padded_box_spec(spec, _spec_eta(spec, eta),
                                      occupation_reach_ry=occupation_reach_ry)
                  for spec in refit]
        # Its own stage: a refit is host work between the W response and the
        # Sigma tau sweep, 2-27 s per CrI3 8x8 SC map (P2-S, 2026-09-25).
        label = "escaped"
        with timing.section("sigma.rule_refit", announce=True,
                            label=f"Sigma rule refit ({len(padded)} {label} windows)"):
            new_fits, fit_rows = fit_sigma_box_specs(
                padded, eta, eps=eps, scope=scope, build_widen=False)
        for spec, padded_spec, fit in zip(refit, padded, new_fits):
            rules[spec["name"]] = _certified_entry(
                dict(fit, rule_source=f"rebuild:sc-fixed:{fit['rule_source']}"),
                padded_spec, spec, rebuilt_at_iteration=iteration,
                rebuild_reason=reasons_by_name[spec["name"]])
    # A product window may temporarily have no live state/pole tuples.  Keep
    # its frozen receipt in ``rules`` and simply omit its zero
    # contribution from this map; if it reappears, the containment check
    # above applies to it again.
    recomputed = dict(reasons_by_name)
    fits = []
    for spec in rows:
        entry = rules[spec["name"]]
        if spec["name"] in reasons_by_name:
            fits.append(dict(entry["fit"]))
            continue
        try:
            fits.append(_fixed_fit_for_spec(entry, spec))
        except _RuleValidityFailure as exc:
            padded_spec = _sc_padded_box_spec(
                spec, _spec_eta(spec, eta), occupation_reach_ry=occupation_reach_ry)
            with timing.section("sigma.rule_refit", announce=True,
                                label="Sigma rule refit (validity)"):
                new_fits, new_rows = fit_sigma_box_specs(
                    [padded_spec], eta, eps=eps,
                    scope=scope, build_widen=False)
            rebuilt = dict(new_fits[0])
            rebuilt["rule_source"] = "rebuild:sc-fixed-validity"
            rules[spec["name"]] = _certified_entry(
                rebuilt, padded_spec, spec, rebuilt_at_iteration=iteration,
                rebuild_reason=f"validity: {exc}")
            recomputed[spec["name"]] = f"validity: {exc}"
            fit_rows.extend(new_rows)
            fits.append(dict(rebuilt))
    if recomputed:
        # Each rebuild is named in the receipt (``sc_fixed_recompute_reasons``),
        # which the Sigma caller prints to the report as one
        # ``SC fixed quadrature recompute:`` line per window.
        session["rebuild_count"] = int(
            session.get("rebuild_count", 0)) + len(recomputed)
    return fits, fit_rows, receipt(
        "frozen", fits, event=("extend" if reasons_by_name else "hold"),
        rebuilt=(name for name in (spec["name"] for spec in rows) if name in recomputed),
        reasons=recomputed.items(), escaped=len(reasons_by_name))


def sigma_box_executor_nodes(
    fit, pole_sign, eta_ry, *, one_sided_hermitian=False,
):
    """Convert one accepted box rule to the dynamic-Sigma executor contract.

    The shared physical convention is

    ``time_exec = pole_sign * time`` and
    ``alpha_exec = weight * exp(-eta * time_exec)``.

    The fitted lower-half-plane valence rule has already received the exact
    ``time=-conj(time), weight=conj(weight)`` transformation in
    :func:`fit_sigma_box_specs`.  ``one_sided_hermitian`` retains PPM's
    crossing channel and its global completion ``(Z-Z^dagger)/(2i)``.  In
    that contract the coefficient is multiplied by ``i`` so completion is
    exactly the Hermitian part of the full causal box sum:

    ``(i Q - (i Q)^dagger)/(2i) = (Q + Q^dagger)/2``.

    No channel is collapsed and the equality holds for general complex box
    times and weights, not only a real one-sided sine grid.
    """
    sign, eta = float(pole_sign), float(eta_ry)
    if sign not in (-1.0, 1.0):
        raise ValueError("Sigma box pole_sign must be +1 or -1")
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("sigma_quadrature requires eta_ry > 0")
    time_exec = sign * np.asarray(fit["times"], np.complex128)
    alpha_exec = (np.asarray(fit["weights"], np.complex128)
                  * np.exp(-eta * time_exec))
    if one_sided_hermitian:
        alpha_exec = 1.0j * alpha_exec
    return MinimaxNodes(
        t=jnp.asarray(time_exec, dtype=jnp.complex128),
        alpha=jnp.asarray(alpha_exec, dtype=jnp.complex128))


def _coarse_runs(base_name, owned, positions, frequencies, omega_eta, omega_grp, fixed,
                 states, pole_stats, pole_sign, eta, eps, held_rules):
    """How one product window serves the coarse (semicore) windows it owns.

    Returns ``([(suffix, owned indices, eta), ...], report)``.  Near samples
    keep the plan's eta.  On a crossing window the coarse windows are grouped
    into runs of adjacent automatic windows of one eta (a user window is a run
    of its own); the grouping minimizes the summed closed-form node count
    (:func:`_law_node_count` of the box each run is built on, the request
    widened and snapped by :func:`_build_box`, at the coarse ``eps``), fewer
    runs on a tie.  On the build box the law is the certified count of 300 of
    311 crossing windows in the 2026-09-28/29 receipts (runs/DEV 584-610); the
    request box missed the 1 % widening, up to 37 nodes on a coarse window.  At SC map 0 the choice is made and the held rule names carry it
    later.  A sign-definite window serves everything at ``eta``.  O(G^2) laws.
    """
    grp = omega_grp[positions[owned]]
    if not np.any(grp >= 0):
        return [("", owned, eta)], None
    near = owned[grp < 0]
    pieces = [("", near, eta)] if near.size else []
    prefix = base_name + "@"
    held = (None if held_rules is None
            else sorted(k[len(prefix):] for k in held_rules if k.startswith(prefix)))
    crossing = make_sigma_box_spec(
        name=base_name, frequencies=frequencies[owned], states=states,
        pole_stats=pole_stats, pole_sign=pole_sign, eta_ry=eta)["kind"] == "crossing"
    if (held is None and not crossing) or (held is not None and not held):
        return [("", owned, eta)], None
    groups = sorted(set(int(g) for g in grp[grp >= 0]))
    eta_of = {g: float(omega_eta[positions[owned[grp == g]]][0]) for g in groups}

    def box_law(run):
        idx = owned[np.isin(grp, run)]
        spec = make_sigma_box_spec(name=base_name, frequencies=frequencies[idx], states=states,
                                   pole_stats=pole_stats, pole_sign=pole_sign,
                                   eta_ry=eta_of[run[0]])
        return _law_node_count(_build_box(spec["box"], eta_of[run[0]], widen=True), eps)

    if held is not None:
        runs = []
        for key in held:
            a, b = (int(x) for x in key.split("g", 1)[1].split("-"))
            run = [g for g in groups if a <= g <= b]
            if run:
                runs.append(run)
        runs += [[g] for g in groups if not any(g in r for r in runs)]
    else:
        n = len(groups)
        best = [(0, 0, None)] + [None] * n          # (law, run count, previous cut)
        for j in range(1, n + 1):
            for i in range(j, 0, -1):
                run = groups[i - 1:j]
                if (len(run) > 1 and (any(fixed[g] for g in run if g < len(fixed))
                                      or len({eta_of[g] for g in run}) > 1
                                      or run[-1] - run[0] != len(run) - 1)):
                    break
                cand = (best[i - 1][0] + box_law(run), best[i - 1][1] + 1, i - 1)
                if best[j] is None or cand[:2] < best[j][:2]:
                    best[j] = cand
        runs, j = [], n
        while j > 0:
            i = best[j][2]
            runs.append(groups[i:j])
            j = i
        runs.reverse()
    for run in runs:
        idx = owned[np.isin(grp, run)]
        pieces.append((f"@eta{eta_of[run[0]] * RYD_TO_EV:.3g}g{run[0]}-{run[-1]}", idx,
                       eta_of[run[0]]))
    report = None
    if held is None:
        one = (box_law(groups) if len({eta_of[g] for g in groups}) == 1
               and not any(fixed[g] for g in groups if g < len(fixed)) else None)
        report = {"window": base_name, "coarse_windows": len(groups),
                  "one_window_law": one,
                  "per_window_law": int(sum(box_law([g]) for g in groups)),
                  "runs": [list(r) for r in runs],
                  "chosen_law": int(sum(box_law(r) for r in runs))}
    return pieces, report


def plan_sigma_windows(
    pole_summaries,
    branches,
    omega_ry,
    eta_ry,
    *,
    eps,
    scope,
    print_fn=print,
    edge_factor=1.5,
    fixed_rule_session=None,
    analytic_line=False,
    material_class=None,
    fixed_pole_support_ry=None,
    certificate_pole_summaries=None,
    occupation_reach_ry=None,
    omega_eta_ry=None,
    omega_group=None,
    group_fixed=None,
):
    """Build the complete MPA Sigma quadrature from raw support boxes.

    ``analytic_line`` asks the service for its fixed-height rule only when
    the box crosses zero and all poles are real. PPM supplies that request;
    generic MPA and shared-pole W keep their box-rule contracts.

    Parameters
    ----------
    pole_summaries
        Concatenated output of ``summarize_sigma_poles``.  Each row contains
        only live per-pole extrema for the all/shallow/deep selectors.
    branches
        Causal ``_SigmaBranch`` rows.  Their masks and optional occupation
        weights are exactly the state support the executor will consume.
    omega_ry
        Requested external frequency grid in Ry.  Used to verify each
        branch's global indices before constructing its denominator corners.
    eta_ry
        Positive retarded broadening in Ry.  It enters both the box's
        imaginary extent and, exactly once, the executor weights.
    eps
        Per-window uniform sup ceiling.  The rule builder certifies directly
        at this value, using relative error on sign-definite boxes and
        peak-relative error on crossing boxes; this matches the measured
        Sigma error currency.
    scope
        The in-process request scope (:func:`sigma_rule_scope`,
        :data:`RUN_SCOPE`) whose earlier rules this plan may reuse, or
        ``None``; nothing in it outlives the process.
    fixed_rule_session
        Mutable run-local receipt used only by a multi-map SC calculation.
        Its first call serves the one-shot rules and certifies the same
        boxes padded by the fixed SC policy; every later map reuses those
        nodes by containment.  ``None``
        preserves the ordinary one-shot planner byte-for-byte.
    fixed_pole_support_ry
        Optional positive real-pole endpoint that the frozen fixed rule
        must cover. The declared ``[0, endpoint]`` interval is intersected
        with each existing pole selector only for the SC certificate; live
        pole statistics, references and executor intervals remain unchanged.
    certificate_pole_summaries
        Optional ``summarize_*`` rows of a pole census that contains
        ``pole_summaries`` (a shared-pole map's union over its sectors).
        They set only each window's certificate box, so every sector call
        of one map requests the same boxes and reuses one set of fits; the
        executor's pole selection, factor references and window kind stay
        this call's own.  A window whose union box would change kind keeps
        its own box.
    occupation_reach_ry
        A metal's occupation floor reach X
        (``efermi.occupation_floor_reach_ry``): no branch state lies past
        ``-X`` in its own coordinate, so the SC plan clips a crossing window's
        inner state pad there.  None (an insulator) leaves the 2 eta pad.
        Used only with ``fixed_rule_session``.
    omega_eta_ry, omega_group, group_fixed
        Optional per-frequency broadening (one value >= ``eta_ry`` per
        ``omega_ry`` sample), coarse-window index (-1 = the near grid) and,
        per index, whether the window is a user's (``gw.qp_support``,
        the SC semicore class).  A crossing product window that owns coarse
        samples serves them in their own windows at their own eta, at
        ``max(eps, qp_support.SEMICORE_EPS)``: :func:`_coarse_runs` groups adjacent automatic windows of
        one eta so that the closed-form node count (:func:`_law_node_count`)
        is least; a user window is never grouped.  A sign-definite product
        window serves every frequency it owns at ``eta_ry`` (its node count
        barely depends on eta).  In an SC run the grouping is decided at map
        0 and held, so a moving pole never adds a window.  State and pole
        edges stay at ``eta_ry``.

    Returns
    -------
    windows, geometry
        Executable ``SharedSigmaWindow`` rows and a JSON-compatible planning
        report.

    Notes
    -----
    Tempting, and why not:

    * Histogram-weight the support: that made a low-mass Fermi state 0.95 meV
      wrong on Na.  Every live tuple gets the same box certificate instead.
    * Merge a whole branch: it replaces three cheap sign-aware boxes by one
      wide crossing box and can silently reintroduce measure dependence.
    * Use peak-relative sup on sign-definite tails: a semicore term at
      ``|d|/eta ~ 200`` then spends about 200 times the intended relative
      error; the builder's relative currency measured 0.1 rather than 4 meV.
    * Retry at tighter ``eps``: a sup, noise, growth, or resource refusal is
      already about this box; a hidden retry is a second accuracy policy.
    * Widen near-zero edges for reuse hits: those edges set crossing rank and
      do not drift; measured 3% all-edge widening added 67 pairs on Na.
    * Reserve another factor for the number of product windows: the windows
      partition the causal ``(state, pole, omega-sign)`` tuples, so every
      denominator error enters Sigma exactly once.  If a box certificate is
      ``|Q(d)-1/d| <= eps/eta`` (and ``<= eps/|d|`` on a sign-definite box),
      then one state's error obeys
      ``|delta Sigma_n| <= sum_p |M_np| eps/eta``.  There is no window-count
      factor.  Lane E's blanket 0.1 reserve raised the Si/Na pair counts from
      551/579 to 690/831 (+25.23%/+43.52%) without measurable accuracy gain.
    """
    started = time.perf_counter()
    eta, tolerance = float(eta_ry), float(eps)
    edge = float(edge_factor)
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("sigma_quadrature requires eta_ry > 0")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("sigma_quadrature_eps must lie in (0, 1)")
    if not np.isfinite(edge) or edge < 0.0:
        raise ValueError("sigma_window_edge_factor must be nonnegative")
    fixed_pole_support = None
    if fixed_pole_support_ry is not None:
        fixed_pole_support = float(fixed_pole_support_ry)
        if (fixed_rule_session is None or not np.isfinite(fixed_pole_support)
                or fixed_pole_support <= 0.0):
            raise ValueError(
                "fixed_pole_support_ry requires a fixed SC session and a "
                "finite positive endpoint")
        previous = fixed_rule_session.setdefault(
            "pole_support_ry", fixed_pole_support)
        # The treatment ceiling only grows (a span past it re-plans it,
        # shared_pole_recipe._sector_treatment_ceiling); every live pole is
        # still box-checked against its held rule.
        if fixed_pole_support < float(previous):
            raise ValueError(
                "fixed SC pole support shrank after initialization: "
                f"{previous!r}->{fixed_pole_support!r} Ry")
        fixed_rule_session["pole_support_ry"] = fixed_pole_support
    branch_rows = list(branches)
    summaries = tuple(pole_summaries)
    if not summaries:
        raise ValueError("Sigma box planning needs at least one pole summary")
    omega_grid = np.asarray(omega_ry, dtype=np.float64)
    omega_eta = None
    if omega_eta_ry is not None:
        omega_eta = np.asarray(omega_eta_ry, dtype=np.float64).reshape(-1)
        if (omega_eta.shape != omega_grid.shape or not np.isfinite(omega_eta).all()
                or np.any(omega_eta < eta * (1.0 - 1.0e-12))):
            raise ValueError("omega_eta_ry must give one finite eta >= eta_ry per frequency")
    coarse_eps = max(tolerance, SEMICORE_EPS)
    omega_grp = (np.full(omega_grid.shape, -1, np.int64) if omega_group is None
                 else np.asarray(omega_group, dtype=np.int64).reshape(-1))
    fixed = tuple(bool(x) for x in (group_fixed or ()))
    split_reports = []
    held_rules = (None if fixed_rule_session is None
                  else fixed_rule_session.get("rules"))
    state_rows, geometry = _product_geometry(branch_rows, eta, edge)

    specs, branch_reports = [], []
    for branch, (state_shape, raw_energy, flat_indices) in zip(
            branch_rows, state_rows):
        positions = np.asarray(branch.omega_idx, dtype=np.int64)
        frequencies = omega_grid[positions]
        expected = (-np.asarray(branch.omega_abs)
                    if branch.neg_omega_half else np.asarray(branch.omega_abs))
        if not np.allclose(frequencies, expected, rtol=0.0, atol=1.0e-13):
            raise ValueError(f"Sigma branch {branch.tag!r} indices disagree")
        pole_sign = 1.0 if branch.space == "cond" else -1.0
        report = {
            "tag": branch.tag, "space": branch.space,
            "negative_frequency_half": bool(branch.neg_omega_half),
            "live_state_count": int(raw_energy.size),
            "plan_start": len(specs), "windows": [],
        }
        omega_abs = np.asarray(branch.omega_abs, np.float64)
        for (name, state_lo, state_hi, selector, pole_lo, pole_hi,
             omega_lo, omega_hi) in _state_products(
                 branch, geometry["state_edge_ry"], geometry["pole_edges_ry"]):
            local = np.nonzero(
                (raw_energy > state_lo) & (raw_energy <= state_hi))[0]
            owned = np.nonzero(
                (omega_abs >= omega_lo) & (omega_abs < omega_hi))[0]
            pole_indices, pole_stats = _pole_rows(summaries, selector)
            if not local.size or not owned.size or not pole_indices.size:
                continue
            states = raw_energy[local]
            owned_all = owned
            base_name = f"{branch.tag}:{name}"
            pieces, split_report = _coarse_runs(
                base_name, owned_all, positions, frequencies, omega_eta, omega_grp, fixed,
                states, pole_stats, pole_sign, eta, coarse_eps, held_rules)
            if split_report is not None:
                split_reports.append(split_report)
            for suffix, owned, eta_w in pieces:
                eta_w = float(eta_w)
                spec = make_sigma_box_spec(
                    name=base_name + suffix, frequencies=frequencies[owned],
                    states=states, pole_stats=pole_stats,
                    pole_sign=pole_sign, eta_ry=eta_w)
                if eta_w != eta:
                    spec["eta_ry"] = eta_w
                if suffix:
                    spec["eps"] = coarse_eps
                if certificate_pole_summaries is not None:
                    _, union_stats = _pole_rows(certificate_pole_summaries, selector)
                    union = (make_sigma_box_spec(
                        name=spec["name"], frequencies=frequencies[owned],
                        states=states,
                        pole_stats=union_stats, pole_sign=pole_sign, eta_ry=eta_w)
                        if union_stats else None)
                    if (union is not None and union["kind"] == spec["kind"]
                            and _box_contains(union["box"], spec["box"])):
                        spec.update(box=union["box"],
                                    raw_real_support=union["raw_real_support"],
                                    pole_extent=union["pole_extent"])
                spec["analytic_line"] = bool(analytic_line)
                if fixed_pole_support is not None:
                    support_lo = max(0.0, float(pole_lo))
                    support_hi = min(fixed_pole_support, float(pole_hi))
                    if support_hi > support_lo:
                        spec["sc_support_pole_extent"] = (
                            support_lo, support_hi, 0.0, 0.0)
                if (fixed_rule_session is not None
                        and name in ("bulk", "state_tail", "pole_tail",
                                     "omega_tail")
                        and geometry["state_edge_ry"] > 0.0
                        and (fixed_pole_support is not None or all(
                            lo >= 0.0 and gamma_lo == gamma_hi == 0.0
                            for lo, _, gamma_lo, gamma_hi in pole_stats))):
                    # The selectors guarantee this gap for positive real poles,
                    # including scalar W without a sector treatment ceiling
                    # (omega_tail: its |omega| cut follows this map's excursion).
                    # Cover future selector members, not only initial samples.
                    spec["sc_selector_gap_ry"] = (
                        _BOX_SIGN_FRACTION * geometry["state_edge_ry"])
                spec.update({
                    "branch": branch,
                    "state_indices": flat_indices[local],
                    "state_shape": state_shape.shape,
                    "state_interval": (float(state_lo), float(state_hi)),
                    "pole_indices": pole_indices,
                    "pole_bounds": (float(pole_lo), float(pole_hi)),
                    "omega_interval": (float(omega_lo), float(omega_hi)),
                    "omega_abs": omega_abs[owned],
                    "omega_idx": positions[owned],
                    "branch_report": report,
                })
                specs.append(spec)
        report["plan_stop"] = len(specs)
        report["window_count"] = report["plan_stop"] - report["plan_start"]
        branch_reports.append(report)

    fixed_receipt = None
    if fixed_rule_session is None:
        fits, fit_rows = fit_sigma_box_specs(
            specs, eta, eps=tolerance, scope=scope)
    else:
        fits, fit_rows, fixed_receipt = _fit_fixed_sc_rules(
            specs, eta, eps=tolerance,
            scope=scope, session=fixed_rule_session,
            material_class=material_class,
            occupation_reach_ry=occupation_reach_ry)
    # The (window, tau) pair count is reported, never refused on: the owner
    # eliminated the pair ceiling (2026-09-02).  A count above what a deck
    # can afford is a planning question answered by eps and the window
    # geometry, not a runtime refusal (TASTE 70).
    pairs = sum(row["node_count"] for row in fits)
    frozen = fixed_receipt is not None and fixed_receipt["mode"] == "frozen"

    output = []
    for spec, fit in zip(specs, fits):
        mask = np.zeros(int(np.prod(spec["state_shape"])), dtype=bool)
        mask[np.asarray(spec["state_indices"], np.int64)] = True
        external_sign = -1 if spec["branch"].neg_omega_half else 1
        window = _SigmaWindow(
            name=spec["name"],
            nodes=sigma_box_executor_nodes(
                fit, spec["pole_sign"], _spec_eta(spec, eta)),
            mask_A=mask.reshape(spec["state_shape"]),
            E_ref_A=spec["E_ref_A"], E_ref_B=spec["E_ref_B"],
            omega_sign=int(spec["pole_sign"]) * external_sign,
            project="full", prefactor=-1.0,
            max_error=fit["sup_error"],
            provenance=(
                f"uniform denominator box {spec['box']}; "
                f"{fit['one_line']}; rule={fit['rule_source']}; "
                f"factor_growth={fit['factor_growth']}"))
        output.append(SharedSigmaWindow(
            window=window, E_A=spec["branch"].E_A,
            omega_abs=spec["omega_abs"], omega_idx=spec["omega_idx"],
            pole_indices=spec["pole_indices"],
            bounds=_pole_bounds(
                len(spec["pole_indices"]), *spec["pole_bounds"]),
            phase_real=np.zeros(len(spec["pole_indices"]), dtype=bool),
            band_weight=spec["branch"].band_weight,
            space=spec["branch"].space))
        spec["branch_report"]["windows"].append({
            "name": spec["name"], "kind": spec["kind"],
            "state_interval_ry": list(spec["state_interval"]),
            "pole_interval_ry": list(spec["pole_bounds"]),
            "omega_abs_interval_ry": list(spec["omega_interval"]),
            "pole_indices": spec["pole_indices"].tolist(),
            "raw_real_support_ry": list(spec["raw_real_support"]),
            "box_ry": list(spec["box"]), "rule_box_ry": list(fit["rule_box"]),
            "node_count": fit["node_count"],
            "node_digest": fit["node_digest"],
            "criterion": ("relative" if fit["relative"]
                          else "peak-relative"),
            "sup_error": fit["sup_error"], "eps": _spec_eps(spec, tolerance),
            "requested_eps": _spec_eps(spec, tolerance),
            "eta_ry": _spec_eta(spec, eta),
            "kappa_max": fit["kappa_max"],
            "roundoff_amplification": fit["roundoff_amplification"],
            "runtime_noise_bound": fit["noise_bound"],
            "runtime_noise_budget": fit["noise_budget"],
            "factor_growth": list(fit["factor_growth"]),
            "rule_source": fit["rule_source"],
            "fit_seconds": fit["seconds"],
            "sc_fixed_rule": frozen,
            "sc_fixed_padded_box_ry": (
                list(fixed_rule_session["rules"][spec["name"]]["padded_box"])
                if frozen else None),
        })
        if _resolve_uniform_rule_trace() and process_rank() == 0:
            print_fn(
                f"[uniform-box] {spec['name']} "
                f"support_re={spec['raw_real_support']} box={spec['box']} (Ry)")

    distinct = sum(len({
        (float(value.real), float(value.imag))
        for row in output[report["plan_start"]:report["plan_stop"]]
        for value in np.asarray(row.window.nodes.t)
    }) for report in branch_reports)
    geometry.update({
        "planner": "uniform_denominator_boxes",
        "eta_ry": eta, "eps": tolerance,
        "rule_eps": tolerance,
        "rule_scope": scope, "rule_family": _RULE_FAMILY,
        "rules_built": sum(1 for row in fit_rows
                           if (row.get("value") or {}).get("built")),
        "n_windows": len(output),
        "window_tau_pairs": pairs, "distinct_tau_count": distinct,
        "plan_seconds": time.perf_counter() - started,
        "planning_process_count": int(process_count()),
        "critical_fit_wall_seconds": max(
            (row["wall_seconds"] for row in fit_rows), default=0.0),
        "branches": branch_reports,
        "coarse_window_split": split_reports,
    })
    if fixed_rule_session is not None:
        geometry.update({
            "sc_fixed_quadrature": True,
            "sc_rule_mode": fixed_receipt["mode"],
            "sc_fixed_iteration": int(fixed_receipt["iteration"]),
            "sc_fixed_initialized": bool(fixed_receipt["initialized"]),
            "sc_fixed_rebuilds_this_iteration": len(
                fixed_receipt.get("rebuilt", ())),
            "sc_fixed_rebuilt_windows": list(fixed_receipt.get("rebuilt", ())),
            "sc_fixed_escaped_windows": int(fixed_receipt.get("escaped", 0)),
            "sc_fixed_total_rebuild_count": int(
                fixed_receipt.get("rebuild_count_total", 0)),
            "sc_fixed_initial_window_tau_pairs": fixed_rule_session.get(
                "initial_window_tau_pairs"),
            "sc_fixed_material_class": fixed_receipt.get("material_class"),
            "sc_fixed_recompute_reasons": dict(
                fixed_receipt.get("recompute_reasons", ())),
            "sc_fixed_class_flip": fixed_receipt.get("class_flip"),
            "sc_fixed_escape_maps_total": int(
                fixed_receipt.get("escape_maps_total", 0)),
            "sc_state_edge_padding_ev": fixed_receipt["pad_ev"],
            "sc_state_edge_padding_fraction": fixed_receipt["pad_fraction"],
            "sc_inner_state_padding_eta": fixed_receipt["inner_pad_eta"],
            "sc_occupation_reach_ry": fixed_receipt["occupation_reach_ry"],
            "sc_plan_event": fixed_receipt["plan_event"],
            "sc_tau_capacity": int(fixed_receipt["tau_capacity"]),
            "sc_pole_extent_padding_fraction": _SC_POLE_PAD_FRACTION,
            "sc_far_pole_factor": _SC_FAR_POLE_FACTOR,
            "sc_fixed_pole_support_ry": fixed_pole_support,
        })
    else:
        geometry["sc_fixed_quadrature"] = False
    # Keep the accepted rule identity in the normal scientific report,
    # including repeated SC planning calls.
    if process_rank() == 0:
        print_fn("Sigma quadrature receipt: " + _receipt_json(geometry))
    return output, geometry


__all__ = [
    "fit_sigma_box_specs",
    "make_sigma_box_spec",
    "plan_sigma_windows",
    "sigma_box_executor_nodes",
    "sigma_rule_scope",
]

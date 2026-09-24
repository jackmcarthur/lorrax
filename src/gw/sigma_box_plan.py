"""Measure-independent denominator-box quadrature for dynamic Sigma(omega).

The public path in this module is deliberately short:

``physical product window -> denominator box -> rule -> executor nodes``.

Pole fields remain distributed.  MPA supplies the bounded per-pole extrema
returned by :func:`gw.mpa.sigma_windows.summarize_sigma_poles`; PPM supplies
the scalar extrema of each exact ``(q, mu, nu)`` pane.  No residue histogram,
sampled lattice, error apportionment, or campaign-wide selection enters the
quadrature.  The box construction, lower-half-plane conjugation, cache policy,
fit guards, and conversion to executor ``(t, alpha)`` live here once for both
routes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import pickle
import time

from ffi import _services

_services.ensure_on_path()

import jax.numpy as jnp
import numpy as np

from common.collectives import (all_gather_processes, gather_to_host,
                                process_count, process_rank)
from common.units import RYD_TO_EV
from gw.minimax_screening import MinimaxNodes
from gw.mpa.sigma_windows import SharedSigmaWindow
from gw.ppm_windows import _SigmaWindow
from gw.scissor import sc_state_pad_ev
from minimax import (
    analytic_line_box_rule,
    UniformRule,
    boundary_samples,
    build_uniform_rule,
    rule_roundoff_amplification,
)


_FACTOR_GROWTH_CAP = 30.0
_RUNTIME_NOISE_EPSILON = 6.0e-8
_RUNTIME_NOISE_SAFETY = 0.05
_SC_POLE_PAD_FRACTION = 0.10
_BOX_SIGN_FRACTION = 0.7
#: SC planner calls served by the one-shot planner before the rule set
#: freezes (owner 2026-09-24, survey A3). Maps 0 and 1 carry the loop's
#: largest motion (Si: every state rigidly +0.39 eV at map 1), and the flat
#: +-2 eV certificate that absorbed it was paid on every later map. The set
#: now freezes at map 1's output with the classification and pole pads only;
#: a later escape refits it (see ``_fit_fixed_sc_rules``).
_SC_ONE_SHOT_CALLS = 2
#: Pad toward zero for a sign-definite SC window: 0.5 of its distance escaped by 1.6% on TaAs 8^3
#: map 1 and 0.25 again at map 2 (semimetal valence state 30 -> 15 -> 6 meV from E_F); 0.05 floors it below 1 meV.
_SC_ZERO_SIDE_CAP = 0.05
_RULE_CACHE_SCHEMA = "sigma-box-ry-v4"


def _rule_digest(rule, noise_amplification):
    """Authenticate the certificate and its complex128 Ry-inverse nodes."""
    identity = hashlib.sha256(json.dumps(
        [_RULE_CACHE_SCHEMA, list(rule.box), float(rule.eps),
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


def sigma_rule_request_cache(directory, identity, poles2, counts, *, eta, eps):
    """Scope shared-pole rules to authenticated current-map physical inputs.

    ``identity`` is the model's energy/occupation/recipe provenance;
    ``poles2`` [Nq,K] in Ry**2 and ``counts`` [Nq] are the small host census.
    Map labels are excluded (:func:`_map_invariant_identity`): equal physical
    inputs on restart and across SC maps share rules, but changed spectra or
    occupations cannot inherit a previous map's plan.
    Domain containment and the executor noise/growth gates still run on hits.
    """
    if directory is None:
        return None
    inputs = _map_invariant_identity(identity)
    digest = hashlib.sha256(json.dumps(
        [_RULE_CACHE_SCHEMA, inputs, float(eta), float(eps)],
        sort_keys=True).encode())
    for row, count in zip(poles2, counts):
        digest.update(np.asarray([count], dtype="<i8").tobytes())
        digest.update(np.asarray(row[:int(count)], dtype="<f8").tobytes())
    return os.path.join(directory, "request_" + digest.hexdigest())


def resolve_sigma_box_cache_dir(setting, input_dir):
    """Resolve the deck's uniform-rule cache spelling beside its input.

    ``"auto"`` selects ``<input_dir>/tmp/sigma_quadrature_rules``;
    ``"off"`` disables the acceleration; any other relative path is resolved
    against ``input_dir``.  A cache is not an accuracy path: every loaded rule
    is still checked for box containment and the requested error currency.
    """
    raw = str(setting).strip()
    if raw.lower() == "off":
        return None
    root = os.path.abspath(input_dir)
    if raw.lower() == "auto":
        return os.path.join(root, "tmp", "sigma_quadrature_rules")
    expanded = os.path.expanduser(raw)
    return (expanded if os.path.isabs(expanded)
            else os.path.join(root, expanded))


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
    return state_rows, {
        "omega_max_ry": omega_max,
        "state_edge_ry": state_edge,
        "pole_edge_ry": omega_max + state_edge + excursion,
        "negative_state_excursion_ry": excursion,
        "edge_factor": float(edge_factor),
    }


def _state_products(branch, state_edge, pole_edge):
    """The sole owner of the three-window Cartesian partition."""
    crossing = ((branch.space == "cond" and not branch.neg_omega_half)
                or (branch.space == "val" and branch.neg_omega_half))
    if crossing:
        return (
            ("resonant", -np.inf, pole_edge, "shallow", 0.0, pole_edge),
            ("state_tail", pole_edge, np.inf,
             "shallow", 0.0, pole_edge),
            ("pole_tail", -np.inf, np.inf,
             "deep", pole_edge, np.inf),
        )
    return (
        ("bulk", state_edge, np.inf, "all", 0.0, np.inf),
        ("resonant", -np.inf, state_edge,
         "shallow", 0.0, pole_edge),
        ("pole_tail", -np.inf, state_edge,
         "deep", pole_edge, np.inf),
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


def _rule_cache_lookup(
    directory, box, eps, relative, *, noise_amplification_cap,
):
    """Return the smallest compatible rule plus any unreadable-path warnings.

    Only ``_RULE_CACHE_SCHEMA`` entries are served, and each is authenticated
    against its stored digest before any compatibility filter reads it.
    Other schemas (including clock-reduced entries written before
    2026-09-11) are ignored with one warning, not deleted.
    """
    if directory is None:
        return None, ()
    warnings = []
    try:
        entries = [name for name in os.listdir(directory)
                   if name.startswith("rule_") and name.endswith(".npz")]
        names = [name for name in entries
                 if name.startswith(f"rule_{_RULE_CACHE_SCHEMA}_")]
        stale = sorted(set(entries) - set(names))
        if stale:
            warnings.append(
                "WARNING sigma quadrature cache schema migration: "
                f"path={os.path.abspath(directory)}; schema={_RULE_CACHE_SCHEMA}; "
                f"ignored {len(stale)} rule file(s) of another schema, first={stale[0]}; "
                "affected windows will be rebuilt. The files are retained "
                "as prior-run evidence.")
    except OSError as exc:
        path = os.path.abspath(directory)
        warnings.append(
            "WARNING sigma quadrature cache lookup failed; rules will be "
            "rebuilt in memory: "
            f"path={path} error={type(exc).__name__}: {exc}")
        return None, tuple(warnings)
    best = None
    for name in sorted(names):
        path = os.path.abspath(os.path.join(directory, name))
        try:
            with np.load(path) as data:
                # Authenticate the stored object before compatibility filtering.
                # A nearby requested eps may reuse this certificate, but is
                # never substituted into the digest of its immutable identity.
                cached_box = tuple(float(value) for value in data["box"])
                rule = UniformRule(
                    times=np.asarray(data["times"]),
                    weights=np.asarray(data["weights"]),
                    box=cached_box, eps=float(data["eps"]),
                    relative=bool(data["relative"]),
                    theta_deg=float(data["theta_deg"]),
                    rank=int(data["rank"]),
                    sup_error=float(data["sup_error"]),
                    kappa_max=float(data["kappa_max"]), seconds=0.0)
                amplification = float(data["roundoff_amplification"])
                if (str(data["schema"]) != _RULE_CACHE_SCHEMA
                        or not _rule_is_certified(rule, rule.eps)
                        or rule.times.ndim != 1 or rule.weights.ndim != 1
                        or not np.isfinite(amplification)
                        or str(data["digest"]) != _rule_digest(rule, amplification)):
                    raise ValueError("GATE sigma_rule_integrity: certificate digest/schema mismatch")
                # A cached certificate above eps, or one built for a looser
                # noise consumer, is not a rule for this request whatever its
                # node count (Na pole-tail, 2026-09-05).
                if (abs(rule.eps - eps) > 1.0e-12 * eps
                        or rule.relative != relative
                        or amplification > noise_amplification_cap
                        or rule.sup_error > eps):
                    continue
                if not (cached_box[0] <= box[0]
                        and cached_box[1] >= box[1]
                        and cached_box[2] <= box[2]
                        and cached_box[3] >= box[3]):
                    continue
                if best is None or rule.node_count < best[0].node_count:
                    best = (rule, name)
        except (EOFError, OSError, KeyError, ValueError) as exc:
            warnings.append(
                "WARNING sigma quadrature cache entry is unreadable and "
                "will not be used: "
                f"path={path} error={type(exc).__name__}: {exc}")
    return best, tuple(warnings)


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


def _rule_cache_store(directory, rule, noise_amplification):
    """Atomically store one immutable box certificate, or return a warning."""
    if directory is None:
        return None
    if not (_rule_is_certified(rule, float(rule.eps))
            and np.isfinite(float(noise_amplification))):
        return ("WARNING sigma quadrature cache store refused an uncertified "
                "or non-finite rule (nothing written)")
    digest = _rule_digest(rule, noise_amplification)
    path = os.path.abspath(os.path.join(
        directory, f"rule_{_RULE_CACHE_SCHEMA}_{digest}.npz"))
    temporary = None
    try:
        os.makedirs(directory, exist_ok=True)
        temporary = f"{path}.{os.getpid()}.tmp"
        with open(temporary, "wb") as handle:
            np.savez(
                handle, schema=_RULE_CACHE_SCHEMA, digest=digest,
                box=np.asarray(rule.box, np.float64),
                eps=float(rule.eps), relative=bool(rule.relative),
                times=rule.times, weights=rule.weights,
                sup_error=float(rule.sup_error),
                kappa_max=float(rule.kappa_max),
                roundoff_amplification=float(noise_amplification),
                theta_deg=float(rule.theta_deg), rank=int(rule.rank),
                seconds=float(rule.seconds))
        os.replace(temporary, path)
    except OSError as exc:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
        # A cache is an acceleration, never a second correctness path, so the
        # accepted rule remains usable. Silence is still wrong: it turns every
        # restart into a cold 100+s plan with no explanation.
        return (
            "WARNING sigma quadrature cache write failed; this accepted rule "
            "will be rebuilt on a later run: "
            f"path={path} error={type(exc).__name__}: {exc}")
    return None


def _cache_build_box(box, eta):
    """Widen only far edges so nearby SC iterations hit by containment."""
    extra = 0.01 * max(box[1] - box[0], eta)
    near = 3.0 * eta
    lo = box[0] - extra if box[0] < -near else box[0]
    hi = box[1] + extra if box[1] > near else box[1]
    return (lo, hi, box[2], box[3] * 1.01)


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


def _fit_rule(spec, eps, cache_dir, eta, *, cache_build_widen=True):
    requested_box = spec["box"]
    # This is exactly the builder's default currency predicate.  It is used
    # here only to search cache metadata; cache misses still leave the choice
    # to build_uniform_rule(relative=None).
    relative = requested_box[0] > 0.0 or requested_box[1] < 0.0
    noise_budget = _RUNTIME_NOISE_SAFETY * eps
    noise_amplification_cap = noise_budget / _RUNTIME_NOISE_EPSILON
    analytic_line = (bool(spec.get("analytic_line")) and not relative
                     and requested_box[2] == requested_box[3]
                     and spec["pole_extent"][2:] == (0.0, 0.0))
    if analytic_line:
        # PPM's real poles make Im(d)=eta exactly.  Ask the analytic service
        # for that line; a cached rectangle rule cannot silently preempt it.
        rule = analytic_line_box_rule(requested_box, eps)
        cached, cache_lookup_warnings = rule, ()
        cache_status = "analytic-line"
    else:
        cached, cache_lookup_warnings = _rule_cache_lookup(
            cache_dir, requested_box, eps, relative,
            noise_amplification_cap=noise_amplification_cap)
        if cached is not None:
            rule, cache_name = cached
            cache_status = f"hit:{cache_name}"
        else:
            build_box = (_cache_build_box(requested_box, eta)
                         if cache_dir is not None and cache_build_widen
                         else requested_box)
            build_kwargs = {}
            if relative:
                # For a sign-definite rule the service's kappa is
                # sum|term|/|Q|, while Sigma's noise amplification is
                # |d|*sum|term|.  The certified relative sup error gives
                # |d Q(d)| <= 1 + eps, so this cap is sufficient for the
                # executor's stricter, eps-scaled noise condition.  Crossing
                # rules use peak-relative term mass instead and retain the
                # service's ordinary cancellation cap.
                build_kwargs["kappa_cap"] = (
                    noise_amplification_cap / (1.0 + eps))
            rule = build_uniform_rule(build_box, eps, **build_kwargs)
            cache_status = "miss" if cache_dir is not None else "off"
        # There is no retry.  The builder takes no clock and no pass count,
        # so a second call with the same inputs returns the same rule; the
        # old 5x-budget retry existed only because the first attempt could
        # have been cut short by a deadline, and there is no deadline to
        # lengthen.  A refusal here is now a statement about the box.

    # ONE ACCEPTANCE ON EVERY PATH.  One-shot, fixed-SC initialization and
    # its rebuilds all require the certified sup error at or below eps; the
    # fixed-SC bypass (enforce_sup_error=False, 2026-09-03) let Na retain a
    # conduction pole-tail rule at 400 x eps in every self-consistent arm.
    if not _rule_is_certified(rule, eps):
        cache_note = ("" if cache_dir is None
                      else f", cache directory {os.path.abspath(cache_dir)}")
        raise RuntimeError(
            f"Sigma box window {spec['name']!r} refused: rule sup error "
            f"{float(rule.sup_error):.6g} exceeds eps={eps:.6g} or the rule "
            f"is not finite ({int(np.asarray(rule.times).size)} nodes on box "
            f"{tuple(round(float(v), 6) for v in rule.box)}, kind "
            f"{spec.get('kind', '?')}, cache={cache_status}{cache_note}"
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
    cache_write_warning = None
    if cached is None:
        # Only executor-acceptable rules enter the shared cache.  In
        # particular, a service-level rule that meets its broad default
        # cancellation cap but misses Sigma's eps-scaled noise cap must not
        # poison every subsequent attempt for this box.
        cache_write_warning = _rule_cache_store(
            cache_dir, rule, noise_amplification)
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
        "cache_status": cache_status, "factor_growth": growth,
        "noise_bound": noise_bound, "noise_budget": noise_budget,
        "roundoff_amplification": noise_amplification,
        "node_digest": node_digest,
        "cache_write_warning": cache_write_warning,
        "cache_lookup_warnings": cache_lookup_warnings,
        "one_line": (f"analytic line: {rule.node_count} nodes, "
                     f"sup {rule.sup_error:.2e} (eps {eps:g})"
                     if analytic_line else rule.one_line()),
    }


def _parallel_fits(specs, worker):
    """Fit independent windows once across ranks and replicate small rules."""
    rank, world = int(process_rank()), int(process_count())
    local = []
    for index in range(rank, len(specs), world):
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
        width = int(np.max(lengths))
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


def fit_sigma_box_specs(
    specs, eta_ry, *, eps, cache_dir, cache_build_widen=True,
):
    """Fit independent route-neutral box specifications across processes.

    The input rows must come from :func:`make_sigma_box_spec`.  This function
    owns the shared cache lookup/build, rule acceptance, lower-half-plane
    conjugation, runtime-noise guard, and factored-growth guard.  It returns
    only small replicated rule receipts; route-specific physical selectors
    stay with the caller.
    """
    rows = list(specs)
    eta, tolerance = float(eta_ry), float(eps)
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("sigma_quadrature requires eta_ry > 0")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("sigma_quadrature_eps must lie in (0, 1)")
    return _parallel_fits(
        rows, lambda index: _fit_rule(
            rows[index], tolerance, cache_dir, eta,
            cache_build_widen=bool(cache_build_widen)))


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


def _sc_padded_box_spec(spec, eta):
    """Return the frozen SC certificate box for one product window.

    Every state may drift by its own classification pad ``sc_state_pad_ev``
    before it is reclassified, so the corners are taken over the padded state
    interval ``[min(E - pad(E)), max(E + pad(E))]``: each real edge moves by
    the pad of the state that sets it. Branch state energies already measure
    distance from mu. Pole drift is covered independently by widening every
    real and imaginary pole extent by ten percent. Nothing else is added; a
    map that leaves this box is an escape and refits the set.

    Tempting, and why not: pad every edge by the window's largest pad. A
    crossing rule's node count follows twice its SHORT side, which the
    frontier state sets, while the largest pad belongs to the farthest state:
    on the Si 101 map-0 boxes the two crossing rules certify at 127 + 135
    nodes that way against 108 + 94 per state (one-shot 86 + 93; the retired
    flat +-2 eV pad on top gave 167 + 161).
    runs/runtime/sigma_quad_20260924/m3_planner.
    """
    a_lo, a_hi, gamma_lo, gamma_hi = spec["pole_extent"]
    frac = _SC_POLE_PAD_FRACTION
    padded_poles = [(
        a_lo - frac * abs(a_lo),
        a_hi + frac * abs(a_hi),
        max(0.0, gamma_lo - frac * abs(gamma_lo)),
        gamma_hi + frac * abs(gamma_hi),
    )]
    # A treatment ceiling is a fixed-domain contract, not a live pole. Keep
    # its declared support separate from the current pole statistics and
    # factor references, and use it only to size the frozen certificate.
    if "sc_support_pole_extent" in spec:
        padded_poles.append(tuple(spec["sc_support_pole_extent"]))
    states = np.asarray(spec["states"], dtype=np.float64)
    pad_ry = sc_state_pad_ev(states * RYD_TO_EV) / RYD_TO_EV
    low, high = int(np.argmin(states - pad_ry)), int(np.argmax(states + pad_ry))
    padded_states = np.asarray(
        [states[low] - pad_ry[low], states[high] + pad_ry[high]])
    pole_box, _, _ = _box_for_window(
        spec["frequencies"], padded_states, padded_poles,
        spec["pole_sign"], eta)
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
    # outside a tail at map 0 can enter it at map 1. Cover the selector's
    # guaranteed sign gap, not the accidental nearest initial sample.
    if "sc_selector_gap_ry" in spec:
        gap = float(spec["sc_selector_gap_ry"])
        if spec["kind"] == "sign_definite_negative":
            box[1] = max(box[1], -gap)
        elif spec["kind"] == "sign_definite_positive":
            box[0] = min(box[0], gap)
    padded = dict(spec)
    padded["box"] = tuple(float(value) for value in box)
    padded["kind"] = (
        "sign_definite_positive" if box[0] > 0.0 else
        "sign_definite_negative" if box[1] < 0.0 else "crossing")
    padded["sc_unpadded_box"] = tuple(spec["box"])
    padded["sc_state_pad_ev"] = (float(pad_ry[low] * RYD_TO_EV),
                                 float(pad_ry[high] * RYD_TO_EV))
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
    fit["cache_status"] = "hit:sc-fixed"
    fit["seconds"] = 0.0
    # A failed write was announced on the iteration that attempted it. Reusing
    # the in-memory fixed rule must not repeat the old warning every SC map.
    fit["cache_write_warning"] = None
    fit["cache_lookup_warnings"] = ()
    return fit


def _fit_fixed_sc_rules(
    specs, eta, *, eps, cache_dir, session, material_class=None,
):
    """One-shot rules for the first maps, then one frozen, padded rule set.

    The first ``_SC_ONE_SHOT_CALLS`` planner calls use the ordinary one-shot
    planner (unpadded, cache-served). The next call freezes a rule set on its
    own boxes padded by :func:`_sc_padded_box_spec`, and the same tau node
    map is reused on every later map while it holds. Three events refit:

    * a window that escapes its certificate box, changes error currency, or
      did not exist when the set froze refits the whole rule set for this map
      (owner 2026-09-22, TaAs semimetal SC); the receipt names every refit
      window and the escaped windows' reasons, and keeps the freezing map's pair
      cost;
    * a metal<->insulator flip re-initializes the set under the new class;
    * a rule-validity failure during reuse (factored-log growth above the
      cap) refits that one window.
    """
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
    if iteration <= _SC_ONE_SHOT_CALLS:
        fits, fit_rows = fit_sigma_box_specs(
            rows, eta, eps=eps, cache_dir=cache_dir)
        return fits, fit_rows, {
            "iteration": iteration, "mode": "one-shot", "initialized": False,
            "rebuilt": (), "recompute_reasons": (),
            "rebuild_count_total": int(session.get("rebuild_count", 0)),
            "material_class": session.get("material_class"),
            "class_flip": session.pop("class_flip", None),
        }
    # Owner 2026-09-22 (TaAs semimetal SC): a window that escapes its frozen
    # box, or a new window, refits the rule set for this map instead of
    # refusing.
    escape_reasons = {}
    if "rules" in session:
        for spec in rows:
            entry = session["rules"].get(spec["name"])
            if entry is None:
                escape_reasons[spec["name"]] = "absent when the rules froze"
                continue
            reasons = _box_escape_reasons(
                entry["fit"]["rule_box"], spec["box"])
            if bool(entry["fit"]["relative"]) != (spec["kind"] != "crossing"):
                reasons.append("absolute/relative error currency changed")
            if reasons:
                escape_reasons[spec["name"]] = "escape: " + "; ".join(reasons)
        if escape_reasons:
            session.pop("rules")
    if "rules" not in session:
        session["eta_ry"] = float(eta)
        session["eps"] = float(eps)
        padded = [_sc_padded_box_spec(spec, eta) for spec in rows]
        fits, fit_rows = fit_sigma_box_specs(
            padded, eta, eps=eps,
            cache_dir=cache_dir, cache_build_widen=False)
        rebuild = bool(escape_reasons)
        status = "rebuild:sc-fixed" if rebuild else "init"
        rules = {}
        for spec, padded_spec, fit in zip(rows, padded, fits):
            frozen = dict(fit)
            frozen["cache_status"] = f"{status}:{fit['cache_status']}"
            rules[spec["name"]] = {
                "fit": frozen,
                "padded_box": tuple(padded_spec["box"]),
                "initial_box": tuple(spec["box"]),
            }
        session["rules"] = rules
        if rebuild:
            # The freezing map's pair cost stays the reference the kept line
            # compares against; every window of the set was refit.
            session["rebuild_count"] = int(
                session.get("rebuild_count", 0)) + len(rows)
            rebuilt = tuple(spec["name"] for spec in rows)
            reasons = tuple(sorted(
                (name, escape_reasons.get(name, "refit with the rule set"))
                for name in rebuilt))
        else:
            session["initial_window_tau_pairs"] = int(sum(
                fit["node_count"] for fit in fits))
            rebuilt, reasons = (), ()
        return [dict(rules[spec["name"]]["fit"]) for spec in rows], fit_rows, {
            "iteration": iteration, "mode": "frozen", "initialized": not rebuild,
            "rebuilt": rebuilt, "recompute_reasons": reasons,
            "rebuild_count_total": int(session.get("rebuild_count", 0)),
            "material_class": session.get("material_class"),
            "class_flip": session.pop("class_flip", None),
        }

    rules = session["rules"]
    # A product window may temporarily have no live state/pole tuples.  Keep
    # its frozen receipt in ``rules`` and simply omit its zero
    # contribution from this map; if it reappears, the containment check
    # above applies to it again.
    fit_rows = []
    recomputed = {}
    fits = []
    for spec in rows:
        entry = rules[spec["name"]]
        try:
            fits.append(_fixed_fit_for_spec(entry, spec))
        except _RuleValidityFailure as exc:
            padded_spec = _sc_padded_box_spec(spec, eta)
            new_fits, new_rows = fit_sigma_box_specs(
                [padded_spec], eta, eps=eps,
                cache_dir=cache_dir, cache_build_widen=False)
            rebuilt = dict(new_fits[0])
            rebuilt["cache_status"] = "rebuild:sc-fixed-validity"
            rules[spec["name"]] = {
                "fit": rebuilt,
                "padded_box": tuple(padded_spec["box"]),
                "initial_box": tuple(spec["box"]),
                "rebuilt_at_iteration": iteration,
                "rebuild_reason": f"validity: {exc}",
            }
            recomputed[spec["name"]] = f"validity: {exc}"
            fit_rows.extend(new_rows)
            fits.append(dict(rebuilt))
    if recomputed:
        session["rebuild_count"] = int(
            session.get("rebuild_count", 0)) + len(recomputed)
        if process_rank() == 0:
            for name, reason in recomputed.items():
                print(f"  [sc-fixed] iteration {iteration}: recomputed the "
                      f"rule for {name!r} ({reason})")
    return fits, fit_rows, {
        "iteration": iteration, "mode": "frozen", "initialized": False,
        "rebuilt": tuple(recomputed),
        "recompute_reasons": tuple(sorted(recomputed.items())),
        "rebuild_count_total": int(session.get("rebuild_count", 0)),
        "material_class": session.get("material_class"),
        "class_flip": None,
    }


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


def plan_sigma_windows(
    pole_summaries,
    branches,
    omega_ry,
    eta_ry,
    *,
    eps,
    cache_dir,
    print_fn=print,
    edge_factor=1.5,
    fixed_rule_session=None,
    analytic_line=False,
    material_class=None,
    fixed_pole_support_ry=None,
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
    cache_dir
        Directory for immutable box-rule certificates, or ``None``.
    fixed_rule_session
        Mutable run-local receipt used only by a multi-map SC calculation.
        Its first ``_SC_ONE_SHOT_CALLS`` calls use the one-shot planner; the
        next call certifies boxes padded by the fixed SC policy, and every
        later map reuses the exact same nodes by containment.  ``None``
        preserves the ordinary one-shot planner byte-for-byte.
    fixed_pole_support_ry
        Optional positive real-pole endpoint that the frozen fixed rule
        must cover. The declared ``[0, endpoint]`` interval is intersected
        with each existing pole selector only for the SC certificate; live
        pole statistics, references and executor intervals remain unchanged.

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
    * Widen near-zero edges for cache hits: those edges set crossing rank and
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
        if float(previous) != fixed_pole_support:
            raise ValueError(
                "fixed SC pole support changed after initialization: "
                f"{previous!r}->{fixed_pole_support!r} Ry")
    branch_rows = list(branches)
    summaries = tuple(pole_summaries)
    if not summaries:
        raise ValueError("Sigma box planning needs at least one pole summary")
    omega_grid = np.asarray(omega_ry, dtype=np.float64)
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
        for (name, state_lo, state_hi, selector,
             pole_lo, pole_hi) in _state_products(
                 branch, geometry["state_edge_ry"], geometry["pole_edge_ry"]):
            local = np.nonzero(
                (raw_energy > state_lo) & (raw_energy <= state_hi))[0]
            pole_indices, pole_stats = _pole_rows(summaries, selector)
            if not local.size or not pole_indices.size:
                continue
            states = raw_energy[local]
            spec = make_sigma_box_spec(
                name=f"{branch.tag}:{name}", frequencies=frequencies,
                states=states, pole_stats=pole_stats,
                pole_sign=pole_sign, eta_ry=eta)
            spec["analytic_line"] = bool(analytic_line)
            if fixed_pole_support is not None:
                support_lo = max(0.0, float(pole_lo))
                support_hi = min(fixed_pole_support, float(pole_hi))
                if support_hi > support_lo:
                    spec["sc_support_pole_extent"] = (
                        support_lo, support_hi, 0.0, 0.0)
            if (fixed_rule_session is not None
                    and name in ("bulk", "state_tail", "pole_tail")
                    and geometry["state_edge_ry"] > 0.0
                    and (fixed_pole_support is not None or all(
                        lo >= 0.0 and gamma_lo == gamma_hi == 0.0
                        for lo, _, gamma_lo, gamma_hi in pole_stats))):
                # The selectors guarantee this gap for positive real poles,
                # including scalar W without a sector treatment ceiling.
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
                "omega_abs": np.asarray(branch.omega_abs, np.float64),
                "omega_idx": positions,
                "branch_report": report,
            })
            specs.append(spec)
        report["plan_stop"] = len(specs)
        report["window_count"] = report["plan_stop"] - report["plan_start"]
        branch_reports.append(report)

    fixed_receipt = None
    if fixed_rule_session is None:
        fits, fit_rows = fit_sigma_box_specs(
            specs, eta, eps=tolerance, cache_dir=cache_dir)
    else:
        fits, fit_rows, fixed_receipt = _fit_fixed_sc_rules(
            specs, eta, eps=tolerance,
            cache_dir=cache_dir, session=fixed_rule_session,
            material_class=material_class)
    if process_rank() == 0:
        announced = set()
        for fit in fits:
            warnings = tuple(fit.get("cache_lookup_warnings", ()))
            write_warning = fit.get("cache_write_warning")
            if write_warning:
                warnings += (write_warning,)
            for warning in warnings:
                if warning not in announced:
                    print_fn(warning)
                    announced.add(warning)
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
                fit, spec["pole_sign"], eta),
            mask_A=mask.reshape(spec["state_shape"]),
            E_ref_A=spec["E_ref_A"], E_ref_B=spec["E_ref_B"],
            omega_sign=int(spec["pole_sign"]) * external_sign,
            project="full", prefactor=-1.0,
            max_error=fit["sup_error"],
            provenance=(
                f"uniform denominator box {spec['box']}; "
                f"{fit['one_line']}; cache={fit['cache_status']}; "
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
            "pole_indices": spec["pole_indices"].tolist(),
            "raw_real_support_ry": list(spec["raw_real_support"]),
            "box_ry": list(spec["box"]), "rule_box_ry": list(fit["rule_box"]),
            "node_count": fit["node_count"],
            "node_digest": fit["node_digest"],
            "criterion": ("relative" if fit["relative"]
                          else "peak-relative"),
            "sup_error": fit["sup_error"], "eps": tolerance,
            "requested_eps": tolerance,
            "kappa_max": fit["kappa_max"],
            "roundoff_amplification": fit["roundoff_amplification"],
            "runtime_noise_bound": fit["noise_bound"],
            "runtime_noise_budget": fit["noise_budget"],
            "factor_growth": list(fit["factor_growth"]),
            "cache_status": fit["cache_status"],
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
        "cache_dir": cache_dir, "rule_cache_schema": _RULE_CACHE_SCHEMA,
        "n_windows": len(output),
        "window_tau_pairs": pairs, "distinct_tau_count": distinct,
        "plan_seconds": time.perf_counter() - started,
        "planning_process_count": int(process_count()),
        "critical_fit_wall_seconds": max(
            (row["wall_seconds"] for row in fit_rows), default=0.0),
        "branches": branch_reports,
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
            "sc_fixed_total_rebuild_count": int(
                fixed_receipt.get("rebuild_count_total", 0)),
            "sc_fixed_initial_window_tau_pairs": fixed_rule_session.get(
                "initial_window_tau_pairs"),
            "sc_fixed_material_class": fixed_receipt.get("material_class"),
            "sc_fixed_recompute_reasons": dict(
                fixed_receipt.get("recompute_reasons", ())),
            "sc_fixed_class_flip": fixed_receipt.get("class_flip"),
            "sc_state_edge_padding_ev": max(
                (float(np.max(sc_state_pad_ev(spec["states"] * RYD_TO_EV)))
                 for spec in specs), default=0.0),
            "sc_pole_extent_padding_fraction": _SC_POLE_PAD_FRACTION,
            "sc_fixed_pole_support_ry": fixed_pole_support,
        })
    else:
        geometry["sc_fixed_quadrature"] = False
    # Keep the accepted rule identity in the normal scientific report,
    # including cache-off and repeated SC planning calls.
    if process_rank() == 0:
        print_fn("Sigma quadrature receipt: " + _receipt_json(geometry))
    return output, geometry


__all__ = [
    "fit_sigma_box_specs",
    "make_sigma_box_spec",
    "plan_sigma_windows",
    "resolve_sigma_box_cache_dir",
    "sigma_box_executor_nodes",
]

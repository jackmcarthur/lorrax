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
from minimax import (
    UniformRule,
    box_samples,
    build_uniform_rule,
    rule_roundoff_amplification,
)


_FACTOR_GROWTH_CAP = 30.0
_RUNTIME_NOISE_EPSILON = 6.0e-8
_RUNTIME_NOISE_SAFETY = 0.05
_SC_STATE_PAD_EV = 2.0
_SC_POLE_PAD_FRACTION = 0.10


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
                                                   0.7 * real_lo)
    hi = real_hi + pad if real_hi >= 0.0 else min(real_hi + pad,
                                                   0.7 * real_hi)
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
    reduction_steps=None,
):
    """Return the smallest compatible rule plus any unreadable-path warnings.

    ``reduction_steps`` is part of the rule's identity: a certificate reduced
    under a deterministic step budget and one reduced under the wall clock
    are different rules for the same box, and a deck that names a step
    budget must never be served a clock-reduced entry (or vice versa).
    """
    if directory is None:
        return None, ()
    warnings = []
    try:
        names = [name for name in os.listdir(directory)
                 if name.endswith(".npz")]
    except OSError as exc:
        path = os.path.abspath(directory)
        warnings.append(
            "WARNING sigma quadrature cache lookup failed; rules will be "
            "rebuilt in memory: "
            f"path={path} error={type(exc).__name__}: {exc}")
        return None, tuple(warnings)
    best = None
    for name in names:
        path = os.path.abspath(os.path.join(directory, name))
        try:
            with np.load(path) as data:
                if (abs(float(data["eps"]) - eps) > 1.0e-12 * eps
                        or bool(data["relative"]) != relative):
                    continue
                cached_steps = (int(data["reduction_steps"])
                                if "reduction_steps" in data else -1)
                if cached_steps != (-1 if reduction_steps is None
                                    else int(reduction_steps)):
                    continue
                # Cache entries pre-dating the executor-noise stamp, or
                # entries built for a looser consumer, are not compatible.
                # A cache hit is an acceleration only; it must never hide a
                # builder attempt that can satisfy the active Sigma gate.
                if ("roundoff_amplification" not in data
                        or float(data["roundoff_amplification"])
                        > noise_amplification_cap):
                    continue
                cached_box = tuple(float(value) for value in data["box"])
                if not (cached_box[0] <= box[0]
                        and cached_box[1] >= box[1]
                        and cached_box[2] <= box[2]
                        and cached_box[3] >= box[3]):
                    continue
                rule = UniformRule(
                    times=np.asarray(data["times"]),
                    weights=np.asarray(data["weights"]),
                    box=cached_box, eps=eps, relative=relative,
                    theta_deg=float(data["theta_deg"]),
                    rank=int(data["rank"]),
                    sup_error=float(data["sup_error"]),
                    kappa_max=float(data["kappa_max"]), seconds=0.0)
                if best is None or rule.node_count < best[0].node_count:
                    best = (rule, name)
        except (EOFError, OSError, KeyError, ValueError) as exc:
            warnings.append(
                "WARNING sigma quadrature cache entry is unreadable and "
                "will not be used: "
                f"path={path} error={type(exc).__name__}: {exc}")
    return best, tuple(warnings)


def _rule_cache_store(directory, rule, noise_amplification,
                      reduction_steps=None):
    """Atomically store one immutable box certificate, or return a warning."""
    if directory is None:
        return None
    steps = -1 if reduction_steps is None else int(reduction_steps)
    identity = hashlib.sha256(json.dumps(
        ["sigma-noise-currency-v1", list(rule.box), float(rule.eps),
         bool(rule.relative), steps]
    ).encode())
    identity.update(np.ascontiguousarray(rule.times).view(np.uint8).tobytes())
    identity.update(np.ascontiguousarray(rule.weights).view(np.uint8).tobytes())
    identity.update(np.asarray(
        [float(rule.sup_error), float(rule.kappa_max),
         float(noise_amplification), float(rule.theta_deg), float(rule.rank)],
        dtype=np.float64).tobytes())
    digest = identity.hexdigest()[:16]
    path = os.path.abspath(os.path.join(directory, f"rule_{digest}.npz"))
    temporary = None
    try:
        os.makedirs(directory, exist_ok=True)
        temporary = f"{path}.{os.getpid()}.tmp"
        with open(temporary, "wb") as handle:
            np.savez(
                handle, box=np.asarray(rule.box, np.float64),
                eps=float(rule.eps), relative=bool(rule.relative),
                times=rule.times, weights=rule.weights,
                sup_error=float(rule.sup_error),
                kappa_max=float(rule.kappa_max),
                roundoff_amplification=float(noise_amplification),
                theta_deg=float(rule.theta_deg), rank=int(rule.rank),
                seconds=float(rule.seconds), reduction_steps=int(steps))
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


def _fit_rule(
    spec, eps, reduction_seconds, cache_dir, eta, *, cache_build_widen=True,
    enforce_sup_error=True, reduction_steps=None,
):
    requested_box = spec["box"]
    # This is exactly the builder's default currency predicate.  It is used
    # here only to search cache metadata; cache misses still leave the choice
    # to build_uniform_rule(relative=None).
    relative = requested_box[0] > 0.0 or requested_box[1] < 0.0
    noise_budget = _RUNTIME_NOISE_SAFETY * eps
    noise_amplification_cap = noise_budget / _RUNTIME_NOISE_EPSILON
    cached, cache_lookup_warnings = _rule_cache_lookup(
        cache_dir, requested_box, eps, relative,
        noise_amplification_cap=noise_amplification_cap,
        reduction_steps=reduction_steps)
    if cached is not None:
        rule, cache_name = cached
        cache_status = f"hit:{cache_name}"
    else:
        build_box = (_cache_build_box(requested_box, eta)
                     if cache_dir is not None and cache_build_widen
                     else requested_box)
        # A step budget replaces the wall clock: deterministic across
        # machines (sigma_quadrature_reduction_steps).
        build_kwargs = (
            {"reduction_steps": int(reduction_steps)}
            if reduction_steps is not None
            else {"time_budget": reduction_seconds})
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

    # Preserve the historical one-shot acceptance policy exactly.  Only the
    # fixed-SC initializer delegates finite-box acceptance to the service: its
    # reduction budget may expire and return the interpolatory eps/10 start,
    # whose ``sup_error`` is a check-cloud diagnostic.  Reapplying this gate
    # there made the time budget a correctness switch and refused the padded
    # Si boxes at 3e-5.
    if enforce_sup_error and rule.sup_error > eps:
        raise RuntimeError(
            f"Sigma box window {spec['name']!r} refused: rule sup error "
            f"{rule.sup_error:.6g} exceeds eps={eps:.6g}")
    # Runtime perturbations must be bounded in the SAME currency as the
    # approximation.  ``kappa = sum|term|/|Q|`` is already relative for a
    # sign-definite box, but it overstates a crossing box's peak-relative
    # error by ~|d|/eta at its far edge.  Measure rho*sum|term| directly.
    noise_cloud = box_samples(
        *rule.box, per_unit=8.0, n_im=48)
    noise_rho = (np.abs(noise_cloud) if rule.relative
                 else float(np.min(noise_cloud.imag)))
    noise_amplification = rule_roundoff_amplification(
        rule.times, rule.weights, noise_cloud, noise_rho)
    noise_bound = noise_amplification * _RUNTIME_NOISE_EPSILON
    if noise_bound > noise_budget:
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
            cache_dir, rule, noise_amplification,
            reduction_steps=reduction_steps)
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
        "one_line": rule.one_line(),
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
    specs, eta_ry, *, eps, reduction_seconds, cache_dir,
    cache_build_widen=True, enforce_sup_error=True, reduction_steps=None,
):
    """Fit independent route-neutral box specifications across processes.

    The input rows must come from :func:`make_sigma_box_spec`.  This function
    owns the shared cache lookup/build, rule acceptance, lower-half-plane
    conjugation, runtime-noise guard, and factored-growth guard.  It returns
    only small replicated rule receipts; route-specific physical selectors
    stay with the caller.
    """
    rows = list(specs)
    eta, tolerance, budget = (
        float(eta_ry), float(eps), float(reduction_seconds))
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("sigma_quadrature requires eta_ry > 0")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("sigma_quadrature_eps must lie in (0, 1)")
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError("sigma_quadrature_reduction_seconds must be > 0")
    return _parallel_fits(
        rows, lambda index: _fit_rule(
            rows[index], tolerance, budget, cache_dir, eta,
            cache_build_widen=bool(cache_build_widen),
            enforce_sup_error=bool(enforce_sup_error),
            reduction_steps=reduction_steps))


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
    """Return the iteration-1 SC certificate box required by policy.

    State drift gets 2 eV on the movable real edge(s).  Pole drift is
    covered independently by widening every real and imaginary pole extent
    by ten percent before recomputing the denominator box.
    """
    a_lo, a_hi, gamma_lo, gamma_hi = spec["pole_extent"]
    frac = _SC_POLE_PAD_FRACTION
    padded_poles = ((
        a_lo - frac * abs(a_lo),
        a_hi + frac * abs(a_hi),
        max(0.0, gamma_lo - frac * abs(gamma_lo)),
        gamma_hi + frac * abs(gamma_hi),
    ),)
    pole_box, _, _ = _box_for_window(
        spec["frequencies"], spec["states"], padded_poles,
        spec["pole_sign"], eta)
    box = [
        min(spec["box"][0], pole_box[0]),
        max(spec["box"][1], pole_box[1]),
        min(spec["box"][2], pole_box[2]),
        max(spec["box"][3], pole_box[3]),
    ]
    state_pad_ry = _SC_STATE_PAD_EV / RYD_TO_EV
    if spec["kind"] in ("crossing", "sign_definite_negative"):
        box[0] -= state_pad_ry
    if spec["kind"] in ("crossing", "sign_definite_positive"):
        box[1] += state_pad_ry
    padded = dict(spec)
    padded["box"] = tuple(float(value) for value in box)
    padded["kind"] = (
        "sign_definite_positive" if box[0] > 0.0 else
        "sign_definite_negative" if box[1] < 0.0 else "crossing")
    padded["sc_unpadded_box"] = tuple(spec["box"])
    padded["sc_state_pad_ev"] = _SC_STATE_PAD_EV
    padded["sc_pole_pad_fraction"] = _SC_POLE_PAD_FRACTION
    if not _box_contains(padded["box"], spec["box"]):
        raise RuntimeError(
            f"SC fixed-rule padding failed to contain {spec['name']!r}")
    return padded


def _fixed_fit_for_spec(entry, spec):
    """Reuse one immutable rule and recheck current factor growth."""
    fit = dict(entry["fit"])
    growth = _factor_growth(
        fit["times"], spec["pole_sign"], spec["states"],
        spec["pole_stats"], spec["E_ref_A"], spec["E_ref_B"])
    if max(growth) > _FACTOR_GROWTH_CAP:
        raise RuntimeError(
            f"Sigma box window {spec['name']!r} refused while reusing its "
            f"fixed SC rule: factored log growth {max(growth):.6g} exceeds "
            f"{_FACTOR_GROWTH_CAP:g}")
    fit["factor_growth"] = growth
    fit["cache_status"] = "hit:sc-fixed"
    fit["seconds"] = 0.0
    # A failed write was announced on the iteration that attempted it. Reusing
    # the in-memory fixed rule must not repeat the old warning every SC map.
    fit["cache_write_warning"] = None
    fit["cache_lookup_warnings"] = ()
    return fit


def _fit_fixed_sc_rules(
    specs, eta, *, eps, reduction_seconds, cache_dir, session,
    reduction_steps=None,
):
    """Fit one padded SC rule set, then reuse those exact nodes.

    Later boxes are checked against their iteration-1 certificates; a
    window that escapes, or a name that did not exist at iteration 1, gets
    its rule rebuilt on the same padded policy and the rebuild is counted
    in the receipt.  Windows that stay inside their certificate reuse the
    exact iteration-1 nodes.
    """
    rows = list(specs)
    iteration = int(session.get("call_count", 0)) + 1
    session["call_count"] = iteration
    if "rules" not in session:
        session["eta_ry"] = float(eta)
        session["eps"] = float(eps)
        padded = [_sc_padded_box_spec(spec, eta) for spec in rows]
        fits, fit_rows = fit_sigma_box_specs(
            padded, eta, eps=eps, reduction_seconds=reduction_seconds,
            cache_dir=cache_dir, cache_build_widen=False,
            enforce_sup_error=False, reduction_steps=reduction_steps)
        rules = {}
        for spec, padded_spec, fit in zip(rows, padded, fits):
            frozen = dict(fit)
            frozen["cache_status"] = f"init:{fit['cache_status']}"
            rules[spec["name"]] = {
                "fit": frozen,
                "padded_box": tuple(padded_spec["box"]),
                "initial_box": tuple(spec["box"]),
            }
        session["rules"] = rules
        session["initial_window_tau_pairs"] = int(sum(
            fit["node_count"] for fit in fits))
        return [dict(rules[spec["name"]]["fit"]) for spec in rows], fit_rows, {
            "iteration": iteration, "initialized": True,
        }

    if (float(session["eta_ry"]) != float(eta)
            or float(session["eps"]) != float(eps)):
        raise ValueError(
            "SC fixed quadrature session changed currency: "
            f"eta {session['eta_ry']!r}->{eta!r}, "
            f"eps {session['eps']!r}->{eps!r}")

    rules = session["rules"]
    # A product window may temporarily have no live state/pole tuples.  Keep
    # its iteration-1 receipt in ``rules`` and simply omit its zero
    # contribution from this map; if it reappears, the same containment and
    # node-identity checks below apply.
    #
    # A genuinely new window name, or a box that escaped its padded
    # certificate, gets its rule REBUILT on the same padded policy as
    # iteration 1 and is counted in the receipt.  Measured 2026-09-03 on
    # production decks (Si 80 bands / 836 centroids, MoS2 ncond=20): the
    # first QP map moves states by 1-3 eV, so a 2 eV certificate taken at
    # the DFT map can be escaped once, and the window set itself changes
    # with the partition.  Refusing there made QSGW unreachable; rebuilding
    # one window's nodes costs one fit and is exactly what the un-frozen
    # loop did every map.  ponytail: rebuilds are unbounded; cap them if a
    # loop is ever seen rebuilding every map.
    rebuild = []
    reasons = {}
    for spec in rows:
        entry = rules.get(spec["name"])
        if entry is None:
            rebuild.append(spec)
            reasons[spec["name"]] = "absent from iteration 1"
            continue
        escaped = _box_escape_reasons(entry["fit"]["rule_box"], spec["box"])
        if escaped:
            rebuild.append(spec)
            reasons[spec["name"]] = "box escape: " + "; ".join(escaped)
    fit_rows = []
    if rebuild:
        padded = [_sc_padded_box_spec(spec, eta) for spec in rebuild]
        new_fits, fit_rows = fit_sigma_box_specs(
            padded, eta, eps=eps, reduction_seconds=reduction_seconds,
            cache_dir=cache_dir, cache_build_widen=False,
            enforce_sup_error=False, reduction_steps=reduction_steps)
        for spec, padded_spec, fit in zip(rebuild, padded, new_fits):
            rebuilt = dict(fit)
            rebuilt["cache_status"] = "rebuild:sc-fixed"
            rules[spec["name"]] = {
                "fit": rebuilt,
                "padded_box": tuple(padded_spec["box"]),
                "initial_box": tuple(spec["box"]),
                "rebuilt_at_iteration": iteration,
                "rebuild_reason": reasons[spec["name"]],
            }
        session["rebuild_count"] = int(session.get("rebuild_count", 0)) + len(rebuild)
        if process_rank() == 0:
            for spec in rebuild:
                print(f"  [sc-fixed] iteration {iteration}: rebuilt the rule for "
                      f"{spec['name']!r} ({reasons[spec['name']]})")
    fits = []
    for spec in rows:
        entry = rules[spec["name"]]
        if entry.get("rebuilt_at_iteration") == iteration:
            fits.append(dict(entry["fit"]))
        else:
            fits.append(_fixed_fit_for_spec(entry, spec))
    return fits, fit_rows, {
        "iteration": iteration, "initialized": False,
        "rebuilt": tuple(spec["name"] for spec in rebuild),
        "rebuild_count_total": int(session.get("rebuild_count", 0)),
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
    reduction_seconds,
    cache_dir,
    print_fn=print,
    edge_factor=1.5,
    fixed_rule_session=None,
    reduction_steps=None,
):
    """Build the complete MPA Sigma quadrature from raw support boxes.

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
    reduction_seconds
        Per-window Gauss-reduction wall budget.  Independent windows are
        assigned round-robin across processes.
    reduction_steps
        ``None`` keeps the wall budget.  An integer replaces it by that
        many reduction passes per window, so the accepted rule -- and the
        Σ τ-node count -- is a function of the inputs alone
        (``sigma_quadrature_reduction_steps``).
    cache_dir
        Directory for immutable box-rule certificates, or ``None``.
    fixed_rule_session
        Mutable run-local receipt used only by a multi-map SC calculation.
        Iteration 1 certifies boxes padded by the fixed SC policy; every
        later map reuses the exact same nodes by containment.  ``None``
        preserves the ordinary one-shot planner byte-for-byte.

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
    budget = float(reduction_seconds)
    edge = float(edge_factor)
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("sigma_quadrature requires eta_ry > 0")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("sigma_quadrature_eps must lie in (0, 1)")
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError("sigma_quadrature_reduction_seconds must be > 0")
    if not np.isfinite(edge) or edge < 0.0:
        raise ValueError("sigma_window_edge_factor must be nonnegative")
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
    if reduction_steps is not None and process_rank() == 0:
        print_fn(
            f"  Sigma rule reduction: deterministic budget of "
            f"{int(reduction_steps)} passes per window "
            "(sigma_quadrature_reduction_seconds ignored)")
    if fixed_rule_session is None:
        fits, fit_rows = fit_sigma_box_specs(
            specs, eta, eps=tolerance, reduction_seconds=budget,
            cache_dir=cache_dir, reduction_steps=reduction_steps)
    else:
        fits, fit_rows, fixed_receipt = _fit_fixed_sc_rules(
            specs, eta, eps=tolerance, reduction_seconds=budget,
            cache_dir=cache_dir, session=fixed_rule_session,
            reduction_steps=reduction_steps)
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
            "sc_fixed_rule": fixed_rule_session is not None,
            "sc_fixed_padded_box_ry": (
                None if fixed_rule_session is None else list(
                    fixed_rule_session["rules"][spec["name"]]["padded_box"])),
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
        "reduction_seconds": budget, "cache_dir": cache_dir,
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
            "sc_fixed_iteration": int(fixed_receipt["iteration"]),
            "sc_fixed_initialized": bool(fixed_receipt["initialized"]),
            "sc_fixed_rebuilds_this_iteration": len(
                fixed_receipt.get("rebuilt", ())),
            "sc_fixed_rebuilt_windows": list(fixed_receipt.get("rebuilt", ())),
            "sc_fixed_total_rebuild_count": int(
                fixed_receipt.get("rebuild_count_total", 0)),
            "sc_fixed_initial_window_tau_pairs": int(
                fixed_rule_session["initial_window_tau_pairs"]),
            "sc_state_edge_padding_ev": _SC_STATE_PAD_EV,
            "sc_pole_extent_padding_fraction": _SC_POLE_PAD_FRACTION,
        })
    else:
        geometry["sc_fixed_quadrature"] = False
    return output, geometry


__all__ = [
    "fit_sigma_box_specs",
    "make_sigma_box_spec",
    "plan_sigma_windows",
    "resolve_sigma_box_cache_dir",
    "sigma_box_executor_nodes",
]

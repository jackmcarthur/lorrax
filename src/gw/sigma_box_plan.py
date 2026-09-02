"""Pole-partitioned denominator-box quadrature for MPA Sigma(omega).

The public path in this module is deliberately short:

``branch -> core/outlier pole products -> denominator box -> rule``.

Pole fields remain distributed.  A cumulative ``abs(B_p)`` census classifies
the outer 1% on either pole-coordinate axis as outliers; ``B_p`` is exactly
the complex residue multiplied by the tau executor.  State occupations are
separate Green-function factors and are not pole weights.  The three normal
Cartesian products use the central pole rectangle.  Each branch also gets
all of its states times its outlier poles, split by denominator sign when
possible.  The selectors are a disjoint cover, so no tuple is dropped.

The CDF is used only to partition poles.  Every product is still fitted by
the measure-independent uniform sup-norm rule on its complete box: there is
no histogram fitting, tuple sampling, error apportionment, or weighted rule
acceptance here.

Tempting, and why not: fitting a residue-weighted histogram made the Fermi
state 0.95 meV wrong (claim 576).  Weight-thresholding tuples and bounding
their omitted contribution is still an exclusion, not a product partition.
Widening every product to every pole is uniform but cost 551 pairs on Si
versus 478 for accepted arm 16 (+15%).  Outlier product windows retain every
pole while preventing tiny-residue edge setters from widening the core.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from common.collectives import (all_gather_processes, gather_to_host,
                                process_count, process_rank)
from gw.minimax_screening import MinimaxNodes
from gw.mpa.sigma_windows import (
    OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
    SharedSigmaWindow,
    summarize_sigma_pole_regions,
    summarize_sigma_poles,
)
from gw.ppm_windows import _SigmaWindow
from minimax import (
    UniformRule,
    box_samples,
    build_uniform_rule,
    rule_amplification_p99,
    rule_sup_error,
)


_FACTOR_GROWTH_CAP = 30.0
_RUNTIME_NOISE_EPSILON = 6.0e-8
_RUNTIME_NOISE_SAFETY = 0.05
_POLE_PERCENTILE_TAIL = 0.01
_POLE_CDF_BINS = 262144
_MAX_SELECTOR_REGIONS = 4


def resolve_sigma_box_cache_dir(setting, input_dir):
    """Resolve the deck's uniform-rule cache spelling beside its input."""
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


def _weighted_pole_rows(summaries, selector):
    """Return pole rows plus their exact executor residue magnitude."""
    indices, stats, weight = [], [], 0.0
    for pole, evidence in summaries:
        row = evidence[selector]
        if row is not None:
            indices.append(int(pole))
            stats.append(tuple(float(value) for value in row[:4]))
            weight += float(row[4])
    return np.asarray(indices, dtype=np.int32), stats, weight


def _region(a_gt=-np.inf, a_le=np.inf, gamma_ge=-np.inf,
            gamma_gt=-np.inf, gamma_lt=np.inf, gamma_le=np.inf):
    return np.asarray(
        (a_gt, a_le, gamma_ge, gamma_gt, gamma_lt, gamma_le),
        dtype=np.float64)


def _fixed_regions(regions):
    """Pad a disjoint selector union to one fixed executor signature."""
    rows = [np.asarray(row, np.float64) for row in regions]
    if len(rows) > _MAX_SELECTOR_REGIONS:
        raise ValueError(
            f"Sigma pole selector needs {len(rows)} regions; fixed capacity "
            f"is {_MAX_SELECTOR_REGIONS}")
    impossible = _region(
        a_gt=np.inf, a_le=-np.inf, gamma_ge=np.inf,
        gamma_gt=np.inf, gamma_lt=-np.inf, gamma_le=-np.inf)
    rows.extend(impossible.copy()
                for _ in range(_MAX_SELECTOR_REGIONS - len(rows)))
    return np.asarray(rows, dtype=np.float64)


def _intersect_a(regions, lower, upper):
    output = []
    for row in regions:
        got = np.asarray(row, np.float64).copy()
        got[0] = max(float(got[0]), float(lower))
        got[1] = min(float(got[1]), float(upper))
        if got[0] < got[1]:
            output.append(got)
    return output


@partial(jax.jit, static_argnames=("bins",))
def _local_pole_cdf(Omega, B, scale, *, bins):
    """Weighted CDF bins on one resident pole shard."""
    a = jnp.real(Omega).reshape(-1)
    gamma = (-jnp.imag(Omega)).reshape(-1)
    residue = jnp.abs(B).reshape(-1)
    live = (jnp.isfinite(a) & jnp.isfinite(gamma)
            & jnp.isfinite(residue) & (residue > 0.0)
            & (a > 0.0) & (gamma >= 0.0))
    weight = jnp.where(live, residue, 0.0)

    def histogram(value):
        unit = jnp.where(live, value / (value + scale), 0.0)
        index = jnp.clip((unit * bins).astype(jnp.int32), 0, bins - 1)
        return jnp.bincount(index, weights=weight, length=bins)

    return histogram(a), histogram(gamma)


def _pole_cdf_batch(Omega, B, eta):
    """Return replicated ``abs(B)`` histograms for one sharded batch."""
    omega_shards = list(Omega.addressable_shards)
    residue_shards = list(B.addressable_shards)
    if len(omega_shards) != 1 or len(residue_shards) != 1:
        raise RuntimeError(
            "Sigma pole CDF expects one addressable pole shard per process")
    local = _local_pole_cdf(
        omega_shards[0].data, residue_shards[0].data,
        float(eta), bins=_POLE_CDF_BINS)
    gathered = [np.asarray(all_gather_processes(
        np.asarray(jax.device_get(row), np.float64)), np.float64)
        for row in local]
    return tuple(np.sum(row, axis=0) for row in gathered)


def _cdf_quantile(histogram, fraction, scale, *, upper_edge):
    """Return a conservative edge of the weighted quantile's CDF bin."""
    histogram = np.asarray(histogram, np.float64)
    cumulative = np.cumsum(histogram, dtype=np.float64)
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Sigma pole CDF has no positive finite residue mass")
    target = float(fraction) * total
    index = min(int(np.searchsorted(cumulative, target, side="left")),
                histogram.size - 1)
    # Keep the complete threshold bin in the core.  Besides making the CDF
    # resolution conservative, this handles an atom correctly: values equal
    # to a percentile are not "beyond" it and must not become outliers.
    unit = (index + int(upper_edge)) / histogram.size
    if unit >= 1.0:
        return np.inf
    return float(scale) * unit / (1.0 - unit)


def _sign_intervals(branch, frequencies, states):
    """Partition pole ``a`` by the sign of the full state product's d."""
    wlo, whi = float(np.min(frequencies)), float(np.max(frequencies))
    elo, ehi = float(np.min(states)), float(np.max(states))
    if branch.space == "cond":
        positive_cut = wlo - ehi
        negative_cut = whi - elo
        return (
            ("negative", negative_cut, np.inf),
            ("crossing", np.nextafter(positive_cut, -np.inf), negative_cut),
            ("positive", -np.inf,
             np.nextafter(positive_cut, -np.inf)),
        )
    negative_cut = -whi - ehi
    positive_cut = -wlo - elo
    return (
        ("negative", -np.inf, np.nextafter(negative_cut, -np.inf)),
        ("crossing", np.nextafter(negative_cut, -np.inf), positive_cut),
        ("positive", positive_cut, np.inf),
    )


def _pole_partition(cdf_a, cdf_gamma, branches, state_rows, omega_grid,
                    geometry, eta):
    """Build the disjoint core/outlier selectors from the 1% pole CDF.

    A branch's real box coordinate is a translation of ``-pole_sign*a``;
    states and requested frequencies therefore do not change pole rank.
    Reversing the order for conduction swaps the 1st and 99th cuts, so the
    symmetric central interval is the same ``a`` interval for both causal
    signs.  The outlier sign split remains branch-specific because its cut
    uses that branch's complete state and frequency extents.  Likewise,
    adding the fixed ``eta`` leaves the ordering of ``gamma`` unchanged.
    """
    tail = _POLE_PERCENTILE_TAIL
    a_lo = _cdf_quantile(cdf_a, tail, eta, upper_edge=False)
    a_hi = _cdf_quantile(cdf_a, 1.0 - tail, eta, upper_edge=True)
    gamma_lo = _cdf_quantile(
        cdf_gamma, tail, eta, upper_edge=False)
    gamma_hi = _cdf_quantile(
        cdf_gamma, 1.0 - tail, eta, upper_edge=True)
    core = [_region(
        a_gt=np.nextafter(a_lo, -np.inf), a_le=a_hi,
        gamma_ge=gamma_lo, gamma_le=gamma_hi)]
    outlier = [
        _region(a_gt=0.0, a_le=np.nextafter(a_lo, -np.inf)),
        _region(a_gt=a_hi),
        _region(a_gt=np.nextafter(a_lo, -np.inf), a_le=a_hi,
                gamma_lt=gamma_lo),
        _region(a_gt=np.nextafter(a_lo, -np.inf), a_le=a_hi,
                gamma_gt=gamma_hi),
    ]
    pole_edge = float(geometry["pole_edge_ry"])
    selectors = {
        "core:all": _fixed_regions(core),
        "core:shallow": _fixed_regions(_intersect_a(
            core, -np.inf, pole_edge)),
        "core:deep": _fixed_regions(_intersect_a(
            core, pole_edge, np.inf)),
    }
    branch_groups = []
    for index, (branch, (_shape, states, _indices)) in enumerate(
            zip(branches, state_rows)):
        frequencies = omega_grid[np.asarray(branch.omega_idx, np.int64)]
        groups = []
        for sign, lower, upper in _sign_intervals(
                branch, frequencies, states):
            regions = _intersect_a(outlier, lower, upper)
            if regions:
                key = f"outlier:{index}:{sign}"
                selectors[key] = _fixed_regions(regions)
                groups.append((key, sign))
        branch_groups.append(groups)
    return {
        "selectors": selectors,
        "branch_groups": branch_groups,
        "a_percentiles_ry": (a_lo, a_hi),
        "gamma_percentiles_ry": (gamma_lo, gamma_hi),
        "tail_fraction_per_side": tail,
        "cdf_bins": _POLE_CDF_BINS,
    }


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


def _rule_cache_lookup(directory, box, eps, relative):
    """Return the smallest cached rule certified on a containing box."""
    if directory is None:
        return None
    try:
        names = [name for name in os.listdir(directory)
                 if name.endswith(".npz")]
    except OSError:
        return None
    best = None
    for name in names:
        try:
            with np.load(os.path.join(directory, name)) as data:
                if (abs(float(data["eps"]) - eps) > 1.0e-12 * eps
                        or bool(data["relative"]) != relative):
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
        except (OSError, KeyError, ValueError):
            continue
    return best


def _rule_cache_store(directory, rule):
    """Atomically store one immutable box certificate."""
    if directory is None:
        return
    try:
        os.makedirs(directory, exist_ok=True)
        digest = hashlib.sha256(json.dumps(
            [list(rule.box), float(rule.eps), bool(rule.relative)]
        ).encode()).hexdigest()[:16]
        path = os.path.join(directory, f"rule_{digest}.npz")
        if os.path.exists(path):
            return
        temporary = f"{path}.{os.getpid()}.tmp"
        with open(temporary, "wb") as handle:
            np.savez(
                handle, box=np.asarray(rule.box, np.float64),
                eps=float(rule.eps), relative=bool(rule.relative),
                times=rule.times, weights=rule.weights,
                sup_error=float(rule.sup_error),
                kappa_max=float(rule.kappa_max),
                theta_deg=float(rule.theta_deg), rank=int(rule.rank),
                seconds=float(rule.seconds))
        os.replace(temporary, path)
    except OSError:
        # A cache is an acceleration, never a second correctness path.
        pass


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


def _fit_rule(spec, eps, reduction_seconds, cache_dir, eta):
    requested_box = spec["box"]
    unbroadened = bool(spec.get("unbroadened", False))
    # A rule fitted on the physical-eta line remains cheap.  PPM's real-axis
    # Laplace limit is a fixed second certificate of those SAME nodes, not a
    # retry: reserve fourfold headroom before measuring the translated line.
    fit_eps = 0.25 * eps if unbroadened else eps
    # This is exactly the builder's default currency predicate.  It is used
    # here only to search cache metadata; cache misses still leave the choice
    # to build_uniform_rule(relative=None).
    relative = requested_box[0] > 0.0 or requested_box[1] < 0.0
    cached = _rule_cache_lookup(
        cache_dir, requested_box, fit_eps, relative)
    if cached is not None:
        rule, cache_name = cached
        cache_status = f"hit:{cache_name}"
    else:
        build_box = (_cache_build_box(requested_box, eta)
                     if cache_dir is not None else requested_box)
        rule = build_uniform_rule(
            build_box, fit_eps, time_budget=reduction_seconds)
        _rule_cache_store(cache_dir, rule)
        cache_status = "miss" if cache_dir is not None else "off"

    times = np.asarray(rule.times, np.complex128)
    weights = np.asarray(rule.weights, np.complex128)
    if rule.sup_error > fit_eps:
        raise RuntimeError(
            f"Sigma box window {spec['name']!r} refused: rule sup error "
            f"{rule.sup_error:.6g} exceeds fit eps={fit_eps:.6g}")
    delivered_error = float(rule.sup_error)
    delivered_kappa = float(rule.kappa_max)
    if unbroadened:
        target = box_samples(
            *requested_box, per_unit=8.0, n_im=12) - 1.0j * eta
        delivered_error, delivered_kappa = rule_sup_error(
            times, weights, target, np.abs(target))
        if delivered_error > eps:
            raise RuntimeError(
                f"Sigma box window {spec['name']!r} refused: translated "
                f"unbroadened relative sup error {delivered_error:.6g} "
                f"exceeds eps={eps:.6g}")
    # The noise clause remains kappa_p99 * 6e-8 <= 0.05 eps, but its
    # percentile is now intrinsic to the certified box: the uniform-rule
    # service area-weights its own fine check cloud.  No physical histogram
    # or sampled tuple lattice is allowed back into rule determination.
    # Tempting, and why not: use the single worst boundary point
    # (rule.kappa_max).  It refused the measured Si crossing rule although
    # its own sup certificate held; one boundary point is not the historical
    # percentile noise contract and cannot be improved by retrying at a
    # tighter eps without reviving the campaign's hidden second stage.
    kappa_p99 = rule_amplification_p99(
        times, weights, requested_box, rule.theta_deg)
    if unbroadened:
        # The max on the exact translated target is stronger than a p99 and
        # prevents the compatibility line from weakening the noise gate.
        kappa_p99 = max(kappa_p99, delivered_kappa)
    noise_bound = kappa_p99 * _RUNTIME_NOISE_EPSILON
    noise_budget = _RUNTIME_NOISE_SAFETY * eps
    if noise_bound > noise_budget:
        raise RuntimeError(
            f"Sigma box window {spec['name']!r} refused: runtime-noise "
            f"bound {noise_bound:.6g} exceeds {noise_budget:.6g}")

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
    return {
        "times": times, "weights": weights,
        "node_count": int(times.size), "rule_box": tuple(rule.box),
        "relative": bool(rule.relative), "sup_error": delivered_error,
        "kappa_p99": kappa_p99, "kappa_max": delivered_kappa,
        "theta_deg": float(rule.theta_deg),
        "rank": int(rule.rank), "seconds": float(rule.seconds),
        "cache_status": cache_status, "factor_growth": growth,
        "noise_bound": noise_bound, "noise_budget": noise_budget,
        "one_line": (rule.one_line()
                     + (f"; translated real-line sup {delivered_error:.2e}"
                        if unbroadened else "")),
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
            spec = specs[index]
            error = (
                f"Sigma box window {spec['name']!r} failed while fitting "
                f"box={tuple(spec['box'])} Ry at kind={spec['kind']!r}: "
                f"{type(exc).__name__}: {exc}")
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


def plan_sigma_windows(
    pole_batches,
    branches,
    omega_ry,
    eta_ry,
    *,
    eps,
    reduction_seconds,
    pair_ceiling,
    cache_dir,
    print_fn=print,
    edge_factor=1.5,
    occupation_window_threshold=OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
    pole_weight_label="abs(B_p)",
    broaden_sign_definite=True,
):
    """Build the MPA Sigma quadrature from core/outlier pole products.

    Parameters
    ----------
    pole_batches
        Zero-argument callable returning a fresh iterator of
        ``(pole_offset, Omega, B)`` resident batches.  The planner walks it
        once for the weighted pole CDF and old boxes, then once for exact
        geometry of the resulting disjoint selectors.
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
        Per-window uniform sup tolerance.  The rule builder uses relative
        error on sign-definite boxes and peak-relative error on crossing
        boxes; this matches the measured Sigma error currency.
    reduction_seconds
        Per-window Gauss-reduction wall budget.  Independent windows are
        assigned round-robin across processes.
    pair_ceiling
        Hard ceiling on the sum of ``(window, tau)`` pairs.
    cache_dir
        Directory for immutable box-rule certificates, or ``None``.
    occupation_window_threshold
        The branch support threshold already used by the executor.  It is
        forwarded unchanged to the old-box census.
    pole_weight_label
        Report label for the pole field supplied as ``B``.  Ordered GN uses
        a branch-neutral witness for both residues; ordinary MPA uses B_p.
    broaden_sign_definite
        Apply the external ``eta`` to sign-definite denominators.  This is
        the MPA convention and the default.  The GN/HL one-pole adapter sets
        it false to retain its established unbroadened Laplace branches;
        crossing boxes still receive ``eta``.  The same box rule is fitted
        with fixed accuracy headroom and explicitly certified after
        translation to the real line.

    Returns
    -------
    windows, geometry
        Executable ``SharedSigmaWindow`` rows and a JSON-compatible planning
        report.

    Notes
    -----
    Tempting, and why not:

    * Histogram-weight the rule fit: that made a low-mass Fermi state 0.95 meV
      wrong on Na.  Pole weights choose products here; every live tuple still
      gets the uniform certificate of the complete product that owns it.
    * Merge a whole branch: it replaces three cheap sign-aware boxes by one
      wide crossing box and can silently reintroduce measure dependence.
    * Use peak-relative sup on sign-definite tails: a semicore term at
      ``|d|/eta ~ 200`` then spends about 200 times the intended relative
      error; the builder's relative currency measured 0.1 rather than 4 meV.
    * Retry at tighter ``eps``: a sup, noise, growth, or resource refusal is
      already about this box; a hidden retry is a second accuracy policy.
    * Widen near-zero edges for cache hits: those edges set crossing rank and
      do not drift; measured 3% all-edge widening added 67 pairs on Na.
    """
    started = time.perf_counter()
    eta, tolerance = float(eta_ry), float(eps)
    budget, ceiling = float(reduction_seconds), int(pair_ceiling)
    edge = float(edge_factor)
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("sigma_quadrature requires eta_ry > 0")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("sigma_quadrature_eps must lie in (0, 1)")
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError("sigma_quadrature_reduction_seconds must be > 0")
    if ceiling < 1:
        raise ValueError("mpa_sigma_max_nodes pair ceiling must be positive")
    if not np.isfinite(edge) or edge < 0.0:
        raise ValueError("sigma_window_edge_factor must be nonnegative")
    branch_rows = list(branches)
    omega_grid = np.asarray(omega_ry, dtype=np.float64)
    state_rows, geometry = _product_geometry(branch_rows, eta, edge)

    summaries = []
    cdf_a = np.zeros(_POLE_CDF_BINS, np.float64)
    cdf_gamma = np.zeros(_POLE_CDF_BINS, np.float64)
    for pole_offset, Omega, B in pole_batches():
        summaries.extend(summarize_sigma_poles(
            Omega, B, branch_rows,
            regularization_width_ry=eta, edge_factor=edge,
            pole_offset=pole_offset,
            occupation_window_threshold=occupation_window_threshold))
        batch_a, batch_gamma = _pole_cdf_batch(Omega, B, eta)
        cdf_a += batch_a
        cdf_gamma += batch_gamma
        del Omega, B
        gc.collect()
    if not summaries:
        raise ValueError("Sigma box planning needs at least one pole batch")
    summaries = tuple(summaries)
    partition = _pole_partition(
        cdf_a, cdf_gamma, branch_rows, state_rows, omega_grid, geometry, eta)
    selected_summaries = []
    for pole_offset, Omega, B in pole_batches():
        selected_summaries.extend(summarize_sigma_pole_regions(
            Omega, B, partition["selectors"], pole_offset=pole_offset))
        del Omega, B
        gc.collect()
    selected_summaries = tuple(selected_summaries)
    total_pole_weight = float(np.sum(cdf_a))
    _core_indices, _core_stats, core_pole_weight = _weighted_pole_rows(
        selected_summaries, "core:all")
    outlier_weight_fraction = max(
        0.0, 1.0 - core_pole_weight / total_pole_weight)
    coverage_residuals = []
    for groups in partition["branch_groups"]:
        covered = core_pole_weight
        for selector_key, _sign in groups:
            _indices, _stats, weight = _weighted_pole_rows(
                selected_summaries, selector_key)
            covered += weight
        residual = abs(covered - total_pole_weight) / total_pole_weight
        if residual > 1.0e-10:
            raise RuntimeError(
                "Sigma pole partition does not cover its residue support: "
                f"relative abs(B_p) residual={residual:.6g}")
        coverage_residuals.append(residual)

    specs, branch_reports = [], []
    for branch_index, (branch, (state_shape, raw_energy, flat_indices)) in (
            enumerate(zip(branch_rows, state_rows))):
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

        def append_spec(name, states, state_indices, state_interval,
                        selector_key, old_selector):
            pole_indices, pole_stats, pole_weight = _weighted_pole_rows(
                selected_summaries, selector_key)
            if not states.size or not pole_indices.size:
                return
            box, raw_real, pole_extent = _box_for_window(
                frequencies, states, pole_stats, pole_sign, eta)
            _old_indices, old_stats = _pole_rows(summaries, old_selector)
            old_box = (_box_for_window(
                frequencies, states, old_stats, pole_sign, eta)[0]
                if old_stats else None)
            kind = ("sign_definite_positive" if box[0] > 0.0 else
                    "sign_definite_negative" if box[1] < 0.0 else
                    "crossing")
            eta_exec = eta
            if kind != "crossing" and not broaden_sign_definite:
                # Fit on the ordinary physical-eta box (cheap rotated ray),
                # but execute and separately certify its real-axis translate.
                # Intrinsic fitted pole widths remain in Omega_p.
                eta_exec = 0.0
            e_ref_a, e_ref_b = _factor_references(
                kind, pole_sign, states, pole_stats)
            specs.append({
                "name": f"{branch.tag}:{name}", "branch": branch,
                "states": states, "state_indices": state_indices,
                "state_shape": state_shape.shape,
                "state_interval": tuple(float(value)
                                        for value in state_interval),
                "pole_indices": pole_indices, "pole_stats": pole_stats,
                "selector_key": selector_key,
                "selector_bounds": partition["selectors"][selector_key],
                "pole_extent": pole_extent, "pole_sign": pole_sign,
                "pole_weight_fraction": pole_weight / total_pole_weight,
                "omega_abs": np.asarray(branch.omega_abs, np.float64),
                "omega_idx": positions, "frequencies": frequencies,
                "raw_real_support": raw_real, "box": box,
                "old_box": old_box, "kind": kind,
                "regularization_width_ry": eta_exec,
                "unbroadened": eta_exec == 0.0,
                "conjugate": pole_sign < 0.0,
                "E_ref_A": e_ref_a, "E_ref_B": e_ref_b,
                "branch_report": report,
            })

        for (name, state_lo, state_hi, selector,
             _pole_lo, _pole_hi) in _state_products(
                 branch, geometry["state_edge_ry"], geometry["pole_edge_ry"]):
            local = np.nonzero(
                (raw_energy > state_lo) & (raw_energy <= state_hi))[0]
            append_spec(
                name, raw_energy[local], flat_indices[local],
                (state_lo, state_hi), f"core:{selector}", selector)
        for selector_key, sign in partition["branch_groups"][branch_index]:
            append_spec(
                f"outlier-{sign}", raw_energy, flat_indices,
                (-np.inf, np.inf), selector_key, "all")
        report["plan_stop"] = len(specs)
        report["window_count"] = report["plan_stop"] - report["plan_start"]
        branch_reports.append(report)

    fits, fit_rows = _parallel_fits(
        specs, lambda index: _fit_rule(
            specs[index], tolerance, budget, cache_dir, eta))
    pairs = sum(row["node_count"] for row in fits)
    if pairs > ceiling:
        raise RuntimeError(
            f"Sigma box plan refused: {pairs} (window,tau) pairs exceed "
            f"mpa_sigma_max_nodes pair ceiling={ceiling}")

    output = []
    for spec, fit in zip(specs, fits):
        time_exec = spec["pole_sign"] * np.asarray(
            fit["times"], np.complex128)
        alpha_exec = (np.asarray(fit["weights"], np.complex128)
                      * np.exp(
                          -spec["regularization_width_ry"] * time_exec))
        mask = np.zeros(int(np.prod(spec["state_shape"])), dtype=bool)
        mask[np.asarray(spec["state_indices"], np.int64)] = True
        external_sign = -1 if spec["branch"].neg_omega_half else 1
        window = _SigmaWindow(
            name=spec["name"],
            nodes=MinimaxNodes(
                t=jnp.asarray(time_exec, dtype=jnp.complex128),
                alpha=jnp.asarray(alpha_exec, dtype=jnp.complex128)),
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
            bounds=np.broadcast_to(
                spec["selector_bounds"],
                (len(spec["pole_indices"]),
                 *spec["selector_bounds"].shape),
            ).copy(),
            phase_real=np.zeros(len(spec["pole_indices"]), dtype=bool),
            band_weight=spec["branch"].band_weight,
            space=spec["branch"].space))
        spec["branch_report"]["windows"].append({
            "name": spec["name"], "kind": spec["kind"],
            "state_interval_ry": list(spec["state_interval"]),
            "pole_selector": spec["selector_key"],
            "pole_regions": spec["selector_bounds"].tolist(),
            "pole_weight_fraction": spec["pole_weight_fraction"],
            "pole_indices": spec["pole_indices"].tolist(),
            "raw_real_support_ry": list(spec["raw_real_support"]),
            "old_box_ry": (None if spec["old_box"] is None
                           else list(spec["old_box"])),
            "box_ry": list(spec["box"]), "rule_box_ry": list(fit["rule_box"]),
            "external_regularization_ry": spec[
                "regularization_width_ry"],
            "node_count": fit["node_count"],
            "criterion": ("relative" if fit["relative"]
                          else "peak-relative"),
            "sup_error": fit["sup_error"], "eps": tolerance,
            "kappa_p99": fit["kappa_p99"],
            "kappa_max": fit["kappa_max"],
            "runtime_noise_bound": fit["noise_bound"],
            "runtime_noise_budget": fit["noise_budget"],
            "factor_growth": list(fit["factor_growth"]),
            "cache_status": fit["cache_status"],
            "fit_seconds": fit["seconds"],
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
        "pole_partition": {
            "weight": str(pole_weight_label),
            "tail_fraction_per_side": partition[
                "tail_fraction_per_side"],
            "cdf_bins": partition["cdf_bins"],
            "cdf_boundary_policy": "retain_complete_threshold_bin",
            "a_percentiles_ry": list(partition["a_percentiles_ry"]),
            "gamma_percentiles_ry": list(
                partition["gamma_percentiles_ry"]),
            "total_pole_weight": total_pole_weight,
            "outlier_weight_fraction": outlier_weight_fraction,
            "branch_coverage_residuals": coverage_residuals,
        },
        "eta_ry": eta, "eps": tolerance,
        "broaden_sign_definite": bool(broaden_sign_definite),
        "reduction_seconds": budget, "cache_dir": cache_dir,
        "pair_ceiling": ceiling, "n_windows": len(output),
        "window_tau_pairs": pairs, "distinct_tau_count": distinct,
        "plan_seconds": time.perf_counter() - started,
        "planning_process_count": int(process_count()),
        "critical_fit_wall_seconds": max(
            (row["wall_seconds"] for row in fit_rows), default=0.0),
        "branches": branch_reports,
    })
    return output, geometry


__all__ = ["plan_sigma_windows", "resolve_sigma_box_cache_dir"]

"""Per-state-mass denominator-box quadrature for MPA Sigma(omega).

The public path in this module is deliberately short:

``branch -> three Cartesian product windows -> denominator box -> rule``.

Pole fields remain distributed.  The planner first retains the bounded
per-pole extrema returned by
:func:`gw.mpa.sigma_windows.summarize_sigma_poles`, then reduces the exact
executor residue magnitude into one small histogram for each of the accepted
``all``/``shallow``/``deep`` pole selectors.  The box geometry and its three
Cartesian product windows are unchanged.  Only the rule's acceptance currency
uses the measure: at every denominator sample it takes the maximum of each
state's own mass, never a sum over states.

For a selected state/pole tuple the executor forms

``d_knp = omega - s_pole * (E_kn + Omega_p) + i eta``

and its scalar coefficient is bounded by

``|M_knp| <= |u_kn| A_kn |B_p(q, mu, nu)| / N_k``.

Here ``u_kn`` is the branch's exact occupation factor, ``B_p`` is the pole
residue multiplied in :func:`gw.ppm_tau_kernel.build_shared_w_tau`, and
``A_kn`` is the Cauchy--Schwarz envelope of the state and final projection
carriers from
:func:`gw.wavefunction_bundle.projected_state_amplitude_envelope`.  The two
orthonormal flat-k factors give ``1/N_k``.  The inequality uses no phase or
cancellation; its common ``1/N_k`` cancels only when the profile is normalized.

Tempting, and why not: summing residue mass over states made the low-total-mass
Fermi state 0.95 meV wrong (claim 576).  Each state's row must constrain its
own denominators.  No tuple is excluded, no box edge moves, and no outlier-pole
product is introduced: claim 618 measured that partition and rejected it.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import time

import jax.numpy as jnp
import numpy as np

from common.collectives import (all_gather_processes, gather_to_host,
                                process_count, process_rank)
from gw.minimax_screening import MinimaxNodes
from gw.mpa.sigma_windows import (OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
                                  SharedSigmaWindow,
                                  summarize_sigma_poles)
from gw.ppm_windows import _SigmaWindow
from gw.sigma_tolerance_profile import (build_tolerance_profile,
                                        profile_grid,
                                        profile_histogram_batch)
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


def _resolve_uniform_rule_trace():
    """Return whether the debug-only box trace was requested."""
    return bool(os.environ.get("LORRAX_UNIFORM_RULE_TRACE"))


def _live_states(branch, amplitude=None):
    """Return the executor's exact support and state-mass bound on host."""
    energy = np.asarray(gather_to_host(branch.E_A), dtype=np.float64)
    base = np.asarray(gather_to_host(branch.base_mask_A), dtype=bool)
    if base.shape != energy.shape:
        base = np.reshape(base, energy.shape)
    occupation = np.ones(energy.shape, dtype=np.float64)
    live = base & np.isfinite(energy)
    if branch.band_weight is not None:
        weight = np.asarray(
            gather_to_host(branch.band_weight), dtype=np.float64
        ).reshape(energy.shape)
        # The branch builder has already applied the deck's occupation-window
        # threshold.  This final nonzero test mirrors the multiplicative
        # executor and prevents an exactly absent state from widening a box.
        live &= np.isfinite(weight) & (np.abs(weight) > 0.0)
        occupation = np.abs(weight)
    if amplitude is None:
        projected = np.ones(energy.shape, dtype=np.float64)
    else:
        projected = np.abs(np.asarray(
            gather_to_host(amplitude), dtype=np.complex128
        )).reshape(energy.shape)
        if not np.all(np.isfinite(projected[live])):
            raise ValueError(
                f"Sigma box branch {branch.tag!r} has nonfinite projected "
                "state amplitudes")
    indices = np.flatnonzero(live.reshape(-1)).astype(np.int32)
    if not indices.size:
        raise ValueError(f"Sigma box branch {branch.tag!r} has no live states")
    state_mass = occupation * projected / int(energy.shape[0])
    return (energy, energy.reshape(-1)[indices], indices,
            state_mass.reshape(-1)[indices])


def _product_geometry(branches, eta, edge_factor, amplitudes):
    omega_max = max(
        (float(np.max(branch.omega_abs)) for branch in branches
         if branch.omega_abs.size), default=0.0)
    excursion = 0.0
    state_rows = []
    for branch, amplitude in zip(branches, amplitudes):
        shape, energy, indices, mass = _live_states(branch, amplitude)
        excursion = max(excursion, -min(float(np.min(energy)), 0.0))
        state_rows.append((shape, energy, indices, mass))
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


def _rule_cache_lookup(directory, box, eps, relative, profile_digest, rho):
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
                cached_profile = (
                    str(data["profile_digest"].item())
                    if "profile_digest" in data.files else "")
                if cached_profile != str(profile_digest or ""):
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
                    kappa_max=float(data["kappa_max"]), seconds=0.0,
                    profiled=bool(profile_digest))
                if rho is not None:
                    check = box_samples(*box, per_unit=8.0, n_im=48)
                    profile_sup, _ = rule_sup_error(
                        rule.times, rule.weights, check, rho(check))
                    if profile_sup > eps:
                        continue
                if best is None or rule.node_count < best[0].node_count:
                    best = (rule, name)
        except (OSError, KeyError, ValueError):
            continue
    return best


def _rule_cache_store(directory, rule, profile_digest):
    """Atomically store one immutable box certificate."""
    if directory is None:
        return
    try:
        os.makedirs(directory, exist_ok=True)
        digest = hashlib.sha256(json.dumps(
            [list(rule.box), float(rule.eps), bool(rule.relative),
             str(profile_digest or "")]
        ).encode()).hexdigest()[:16]
        path = os.path.join(directory, f"rule_{digest}.npz")
        if os.path.exists(path):
            return
        temporary = f"{path}.{os.getpid()}.tmp"
        with open(temporary, "wb") as handle:
            np.savez(
                handle, box=np.asarray(rule.box, np.float64),
                eps=float(rule.eps), relative=bool(rule.relative),
                profile_digest=str(profile_digest or ""),
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


def _fit_rule(spec, eps, reduction_seconds, cache_dir, eta,
              profile_grid_nodes):
    requested_box = spec["box"]
    # This is exactly the builder's default currency predicate.  It is used
    # here only to search cache metadata; cache misses still leave the choice
    # to build_uniform_rule(relative=None).
    relative = requested_box[0] > 0.0 or requested_box[1] < 0.0
    u_nodes, v_nodes = profile_grid_nodes
    rho, profile_digest, profile_report = build_tolerance_profile(
        requested_box, spec["kind"], spec["pole_sign"], spec["states"],
        spec["state_masses"], spec["frequencies"],
        spec["pole_histogram"], u_nodes, v_nodes, eta, eps)
    cached = _rule_cache_lookup(
        cache_dir, requested_box, eps, relative, profile_digest, rho)
    if cached is not None:
        rule, cache_name = cached
        cache_status = f"hit:{cache_name}"
    else:
        build_box = (_cache_build_box(requested_box, eta)
                     if cache_dir is not None else requested_box)
        rule = build_uniform_rule(
            build_box, eps, time_budget=reduction_seconds,
            relative=relative, rho=rho)
        _rule_cache_store(cache_dir, rule, profile_digest)
        cache_status = "miss" if cache_dir is not None else "off"

    times = np.asarray(rule.times, np.complex128)
    weights = np.asarray(rule.weights, np.complex128)
    if rule.sup_error > eps:
        raise RuntimeError(
            f"Sigma box window {spec['name']!r} refused: rule sup error "
            f"{rule.sup_error:.6g} exceeds eps={eps:.6g}")
    uniform_check = box_samples(*requested_box, per_unit=8.0, n_im=48)
    incumbent_rho = (np.abs(uniform_check) if relative else
                     np.full(uniform_check.shape, requested_box[2]))
    uniform_sup_error, _ = rule_sup_error(
        times, weights, uniform_check, incumbent_rho)
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
        "relative": bool(rule.relative), "sup_error": float(rule.sup_error),
        "kappa_p99": kappa_p99, "kappa_max": float(rule.kappa_max),
        "theta_deg": float(rule.theta_deg),
        "rank": int(rule.rank), "seconds": float(rule.seconds),
        "cache_status": cache_status, "factor_growth": growth,
        "noise_bound": noise_bound, "noise_budget": noise_budget,
        "one_line": rule.one_line(),
        "profile_digest": profile_digest,
        "profile": profile_report,
        "uniform_sup_error": float(uniform_sup_error),
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
    state_amplitudes_by_branch=None,
):
    """Build the MPA Sigma quadrature from profiled raw support boxes.

    Parameters
    ----------
    pole_batches
        Zero-argument callable returning a fresh iterator of
        ``(pole_offset, Omega, B)`` resident batches.  One walk obtains the
        exact old-box extrema and a second reduces the profile histograms;
        neither walk gathers a pole field.
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
        forwarded unchanged to the box census.
    state_amplitudes_by_branch
        Per-branch Cauchy--Schwarz projection envelopes with the same shape
        as ``branch.E_A``.  ``None`` uses unit envelopes for synthetic tests;
        production supplies the wavefunction-derived envelope.

    Returns
    -------
    windows, geometry
        Executable ``SharedSigmaWindow`` rows and a JSON-compatible planning
        report.

    Notes
    -----
    Tempting, and why not:

    * Sum state histograms: that made a low-mass Fermi state 0.95 meV wrong
      on Na.  The profile takes the pointwise maximum of state-owned rows.
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
    if state_amplitudes_by_branch is None:
        amplitude_rows = [None] * len(branch_rows)
    else:
        amplitude_rows = list(state_amplitudes_by_branch)
        if len(amplitude_rows) != len(branch_rows):
            raise ValueError(
                "state_amplitudes_by_branch must have one row per branch")
    omega_grid = np.asarray(omega_ry, dtype=np.float64)
    state_rows, geometry = _product_geometry(
        branch_rows, eta, edge, amplitude_rows)

    summaries = []
    for pole_offset, Omega, B in pole_batches():
        summaries.extend(summarize_sigma_poles(
            Omega, B, branch_rows,
            regularization_width_ry=eta, edge_factor=edge,
            pole_offset=pole_offset,
            occupation_window_threshold=occupation_window_threshold))
        del Omega, B
        gc.collect()
    if not summaries:
        raise ValueError("Sigma box planning needs at least one pole batch")
    summaries = tuple(summaries)
    all_stats = [evidence["all"] for _pole, evidence in summaries
                 if evidence["all"] is not None]
    profile_grid_nodes = profile_grid(
        max(row[1] for row in all_stats),
        max(row[3] for row in all_stats), eta, tolerance)
    selectors = {
        "all": _pole_bounds(1, 0.0, np.inf),
        "shallow": _pole_bounds(
            1, 0.0, geometry["pole_edge_ry"]),
        "deep": _pole_bounds(
            1, geometry["pole_edge_ry"], np.inf),
    }
    profile_histograms = {
        name: np.zeros(
            (len(profile_grid_nodes[0]), len(profile_grid_nodes[1])),
            dtype=np.float64)
        for name in selectors
    }
    for _pole_offset, Omega, B in pole_batches():
        batch_histograms = profile_histogram_batch(
            Omega, B, selectors, *profile_grid_nodes, eta)
        for name, histogram in batch_histograms.items():
            profile_histograms[name] += histogram
        del Omega, B
        gc.collect()

    specs, branch_reports = [], []
    for branch, (state_shape, raw_energy, flat_indices,
                 raw_state_mass) in zip(
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
            box, raw_real, pole_extent = _box_for_window(
                frequencies, states, pole_stats, pole_sign, eta)
            kind = ("sign_definite_positive" if box[0] > 0.0 else
                    "sign_definite_negative" if box[1] < 0.0 else
                    "crossing")
            e_ref_a, e_ref_b = _factor_references(
                kind, pole_sign, states, pole_stats)
            spec = {
                "name": f"{branch.tag}:{name}", "branch": branch,
                "states": states, "state_indices": flat_indices[local],
                "state_masses": raw_state_mass[local],
                "state_shape": state_shape.shape,
                "state_interval": (float(state_lo), float(state_hi)),
                "pole_indices": pole_indices, "pole_stats": pole_stats,
                "pole_bounds": (float(pole_lo), float(pole_hi)),
                "pole_histogram": profile_histograms[selector],
                "pole_extent": pole_extent, "pole_sign": pole_sign,
                "omega_abs": np.asarray(branch.omega_abs, np.float64),
                "omega_idx": positions, "frequencies": frequencies,
                "raw_real_support": raw_real, "box": box, "kind": kind,
                "conjugate": pole_sign < 0.0,
                "E_ref_A": e_ref_a, "E_ref_B": e_ref_b,
                "branch_report": report,
            }
            specs.append(spec)
        report["plan_stop"] = len(specs)
        report["window_count"] = report["plan_stop"] - report["plan_start"]
        branch_reports.append(report)

    fits, fit_rows = _parallel_fits(
        specs, lambda index: _fit_rule(
            specs[index], tolerance, budget, cache_dir, eta,
            profile_grid_nodes))
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
                      * np.exp(-eta * time_exec))
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
                f"per-state-mass denominator box {spec['box']}; "
                f"{fit['one_line']}; cache={fit['cache_status']}; "
                f"factor_growth={fit['factor_growth']}"))
        output.append(SharedSigmaWindow(
            window=window, E_A=spec["branch"].E_A,
            omega_abs=spec["omega_abs"], omega_idx=spec["omega_idx"],
            pole_indices=spec["pole_indices"],
            bounds=_pole_bounds(
                len(spec["pole_indices"]), *spec["pole_bounds"]),
            phase_real=np.zeros(len(spec["pole_indices"]), dtype=bool),
            band_weight=spec["branch"].band_weight))
        spec["branch_report"]["windows"].append({
            "name": spec["name"], "kind": spec["kind"],
            "state_interval_ry": list(spec["state_interval"]),
            "pole_interval_ry": list(spec["pole_bounds"]),
            "pole_indices": spec["pole_indices"].tolist(),
            "raw_real_support_ry": list(spec["raw_real_support"]),
            "box_ry": list(spec["box"]), "rule_box_ry": list(fit["rule_box"]),
            "node_count": fit["node_count"],
            "criterion": ("relative" if fit["relative"]
                          else "peak-relative") + ":per-state-mass-profile",
            "sup_error": fit["sup_error"], "eps": tolerance,
            "uniform_recheck_sup_error": fit["uniform_sup_error"],
            "profile_digest": fit["profile_digest"],
            "profile": fit["profile"],
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
        "planner": "per_state_mass_profiled_denominator_boxes",
        "eta_ry": eta, "eps": tolerance,
        "profile_grid_shape": [
            len(profile_grid_nodes[0]), len(profile_grid_nodes[1])],
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


__all__ = ["plan_sigma_windows"]

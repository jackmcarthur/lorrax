"""Plan delivered-error Sigma quadrature on measured product windows.

The planner has two numerical dials: the delivered envelope target and the
retarded broadening ``eta``.  It divides each causal branch into a few
``state interval x pole interval`` windows, measures their weighted reciprocal
problems, and assigns each window part of the global error budget.

Sign-definite windows fit Hackbusch-seeded noncrossing rules on demand.  Their
acceptance chain is: achieved error on the fitting lattice, achieved error on
the refined validation lattice, and the runtime-noise gate.  An off-axis rule
that misses those gates falls back to the general measure-adapted fitter.
Crossing windows are fitted on demand against their measured pole distribution.
Every fitted rule must also pass the bounded-factor gate.  The planner never
evaluates explicit state--pole pairs.

The final acceptance test is

``kappa_p99 * RUNTIME_NOISE_EPSILON <= AMPLIFICATION_NOISE_SAFETY * target``.

The plan also refuses exponent growth above its bounded-factor limit and more
than 200 total ``(window, tau)`` pairs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
import pickle
import time
import weakref

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from common.collectives import (all_gather_processes, gather_to_host,
                                process_count, process_rank, psum_replicate)
from gw.minimax_screening import MinimaxNodes
from gw.mpa.sigma_windows import SharedSigmaWindow
from gw.ppm_windows import _SigmaBranch, _SigmaWindow
from minimax import ReciprocalMeasureProblem, tail_refined_lattice_measure


DEFAULT_LATTICE_BINS = 25
ENVELOPE_ERROR_SAFETY = 0.8
FACTOR_GROWTH_CAP = 30.0
RUNTIME_NOISE_EPSILON = 6.0e-8
AMPLIFICATION_NOISE_SAFETY = 0.05
#: Overshoot the tightening slightly, because a re-fit at a given allowance
#: lands somewhere at or under it rather than exactly on it.
TIGHTEN_MARGIN = 0.9
#: A single tightening pass never asks for more than this factor at once.  The
#: guaranteed-feasible allowance is the budget split N ways, so no honest deck
#: needs a bigger step than that; anything demanding one is refusing for a
#: reason tightening will not fix, and should say so rather than grind.
TIGHTEN_FLOOR = 0.05
# Default pair budget for the shipped 0..5 eV demonstration grid; the deck's
# max_nodes OWNS the budget (a 4x-wider omega request physically needs more
# crossing nodes — growth is linear in crossing bandwidth). 200 remains the
# default via the config default; it is a resource certificate, not an
# accuracy dial (dial census 2026-08-31, DERIVE).
MAX_WINDOW_TAU_PAIRS = 200
_PLAN_CACHE_VERSION = 8

# The first sign-definite rank passing the consumer gates is the loosest legal
# rule.  One additional rank is cheap for the logarithmic noncrossing family
# and preserves a measured accuracy margin without adding a deck dial.
SIGN_DEFINITE_ACCEPTANCE_RANK_MARGIN = 1

_WINDOW_CENSUS_PREFIX = "[delivered-planner-window] "


class _ProductWindowRefusal(RuntimeError):
    """A product-window refusal with its best measured rule metrics."""

    def __init__(self, message, residual=None, amplification_p99=None):
        super().__init__(message)
        self.residual = residual
        self.amplification_p99 = amplification_p99


class _BudgetShortfall(RuntimeError):
    """Rules exist and fit under the node ceiling, but their costs overshoot.

    Distinct from a refusal: nothing is unservable here, the plan is merely
    priced above the global delivered-error budget.  Carries the numbers the
    planner needs to tighten by exactly the amount it is short.
    """

    def __init__(self, message, best_cost, budget):
        super().__init__(message)
        self.best_cost = float(best_cost)
        self.budget = float(budget)


def _planner_work_indices(count, rank=None, world=None):
    """Return this process's deterministic share of indexed planner work."""
    world = int(process_count() if world is None else world)
    rank = int(process_rank() if rank is None else rank)
    if world < 1 or not 0 <= rank < world:
        raise ValueError(
            f"invalid planner process coordinate rank={rank}, world={world}")
    return tuple(range(rank, int(count), world))


def _all_gather_planner_payload(payload):
    """Gather one variable-length Python payload without creating a mesh.

    The existing process collective accepts fixed-shape arrays.  A length
    gather followed by one padded byte gather carries the small fitted-rule
    dictionaries while preserving NumPy dtypes and every bit of their nodes
    and weights.
    """
    encoded = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    local = np.frombuffer(encoded, dtype=np.uint8)
    lengths = np.asarray(all_gather_processes(
        np.asarray(local.size, dtype=np.int32)), dtype=np.int64).reshape(-1)
    if lengths.size != int(process_count()) or np.any(lengths <= 0):
        raise RuntimeError("planner payload length gather returned bad data")
    width = int(np.max(lengths))
    padded = np.zeros(width, dtype=np.uint8)
    padded[:local.size] = local
    gathered = np.asarray(all_gather_processes(padded), dtype=np.uint8)
    if gathered.shape != (int(process_count()), width):
        raise RuntimeError(
            "planner payload gather returned shape "
            f"{gathered.shape}, expected {(int(process_count()), width)}")
    return [pickle.loads(np.ascontiguousarray(
        gathered[source, :int(length)]).tobytes())
        for source, length in enumerate(lengths)]


def _assemble_planner_rows(shards, count, world, *, refuse_errors):
    """Reassemble indexed work and choose one rank-independent refusal."""
    rows = [row for shard in shards for row in shard]
    rows.sort(key=lambda row: int(row["index"]))
    indices = [int(row["index"]) for row in rows]
    expected = list(range(int(count)))
    if indices != expected:
        raise RuntimeError(
            f"planner gather returned work indices {indices}, expected "
            f"{expected}")
    for row in rows:
        expected_source = int(row["index"]) % int(world)
        if int(row["source_rank"]) != expected_source:
            raise RuntimeError(
                f"planner work {row['index']} came from rank "
                f"{row['source_rank']}, expected {expected_source}")
    refusals = [row for row in rows if row["error"] is not None]
    if refusals and refuse_errors:
        # Rows are index-sorted, so every process raises the same first
        # refusal even when several independent windows fail.
        raise RuntimeError(refusals[0]["error"]["message"])
    values = [None if row["error"] is not None else row["value"]
              for row in rows]
    return values, rows


def _run_parallel_planner_jobs(count, worker, *, refuse_errors):
    """Run indexed fits once across processes, then replicate their results.

    Exceptions are data until after the gather.  This is essential on a
    fail-fast launcher: raising on one process before its peers enter the
    collective would turn an ordinary product-window refusal into a hang.
    """
    rank = int(process_rank())
    world = int(process_count())
    local_rows = []
    for index in _planner_work_indices(count, rank, world):
        started = time.perf_counter()
        try:
            value, detail = worker(index)
            error = None
        except Exception as exc:  # noqa: BLE001 - refusals cross ranks as data
            value, detail = None, {}
            error = {
                "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "message": str(exc),
                "best_achieved_residual": getattr(exc, "residual", None),
                "best_achieved_kappa_p99": getattr(
                    exc, "amplification_p99", None),
            }
        local_rows.append({
            "index": int(index),
            "source_rank": rank,
            "value": value,
            "error": error,
            "seconds": time.perf_counter() - started,
            "detail": detail,
        })
    shards = _all_gather_planner_payload(local_rows)
    return _assemble_planner_rows(
        shards, count, world, refuse_errors=refuse_errors)


def _raise_first_planner_refusal(rows):
    """Raise the first index-ordered planner refusal after diagnostics."""
    refusal = next((row for row in rows if row["error"] is not None), None)
    if refusal is not None:
        raise RuntimeError(refusal["error"]["message"])


def _window_census_geometry(spec, eta):
    """Measure the support coordinates that can change between SC calls."""
    denominator = np.asarray(spec["problem"].denominators, np.complex128)
    magnitude = np.abs(denominator)
    scale_span = float(np.max(magnitude) / np.min(magnitude))
    if spec["kind"] != "crossing":
        return None, None, None, None, scale_span
    oriented = float(spec["pole_sign"]) * denominator
    gamma_min = float(np.min(oriented.imag))
    radius = float(np.max(np.abs(oriented.real)))
    return (radius, gamma_min, radius / float(eta),
            radius / gamma_min, scale_span)


def _emit_window_census(specs, eta, apportioned_targets, *,
                        candidates_by_window=None, planner_rows=None,
                        source):
    """Print one stable JSON record for every fitted product window.

    The mass share uses each window's maximum inverse-gap delivered envelope,
    the same quantity that converts its relative target to the global absolute
    budget.  Only rank zero prints because all planner ranks hold the same
    gathered rows and measured window problems.
    """
    if process_rank() != 0:
        return
    total_mass = float(sum(float(spec["envelope"]) for spec in specs))
    rows = planner_rows if planner_rows is not None else [None] * len(specs)
    candidates = (candidates_by_window if candidates_by_window is not None
                  else [None] * len(specs))
    for spec, apportioned_target, window_candidates, row in zip(
            specs, apportioned_targets, candidates, rows):
        error = None if row is None else row["error"]
        best_residual = (None if error is None else
                         error.get("best_achieved_residual"))
        best_kappa = (None if error is None else
                      error.get("best_achieved_kappa_p99"))
        family = None
        if error is None and window_candidates:
            best = min(
                window_candidates,
                key=lambda candidate: (
                    float(candidate["metrics"][0]),
                    float(candidate["metrics"][1])))
            best_residual = float(best["metrics"][0])
            best_kappa = float(best["metrics"][1])
            family = best["evidence"]["family"]
        radius, gamma_min, A_over_eta, A_over_gamma, span = (
            _window_census_geometry(spec, eta))
        record = {
            "A_over_eta": A_over_eta,
            "A_over_gamma_min": A_over_gamma,
            "apportioned_target": float(apportioned_target),
            "best_achieved_kappa_p99": best_kappa,
            "best_achieved_residual": best_residual,
            "candidate_family": family,
            "cell_count": int(spec["problem"].internal_sums.size),
            "crossing_radius_ry": radius,
            "delivered_mass_share": float(spec["envelope"]) / total_mass,
            # The SHARE is normalised, so it moves when ANY window's mass
            # moves.  Comparing two SC maps needs the absolute envelope
            # and the total beside it, or a window that never changed
            # reads as having gained mass.
            "delivered_envelope": float(spec["envelope"]),
            "delivered_envelope_total": float(total_mass),
            "gamma_min_ry": gamma_min,
            "kind": spec["kind"],
            "name": spec["name"],
            "scale_span": span,
            "source": str(source),
            "status": "refused" if error is not None else "served",
        }
        print(_WINDOW_CENSUS_PREFIX + json.dumps(
            record, sort_keys=True, separators=(",", ":")), flush=True)


def _planner_rows_profile(rows):
    """Summarize fit costs on the rank that determined stage wall time."""
    world = max(
        int(process_count()),
        max((int(row["source_rank"]) + 1 for row in rows), default=1))
    by_rank = {rank: {
        "fit": 0.0, "adapted": 0.0, "shipped": 0.0, "jobs": 0,
    } for rank in range(world)}
    for row in rows:
        bucket = by_rank[int(row["source_rank"])]
        bucket["fit"] += float(row["seconds"])
        bucket["adapted"] += float(
            row["detail"].get("adapted_fit_seconds", 0.0))
        bucket["shipped"] += float(
            row["detail"].get("shipped_fallback_seconds", 0.0))
        bucket["jobs"] += 1
    critical_rank = max(
        by_rank, key=lambda rank: (by_rank[rank]["fit"], -rank), default=0)
    critical = by_rank.get(critical_rank, {
        "fit": 0.0, "adapted": 0.0, "shipped": 0.0, "jobs": 0})
    return {
        "critical_rank": int(critical_rank),
        "critical_rank_fit_seconds": float(critical["fit"]),
        "critical_rank_adapted_fit_seconds": float(critical["adapted"]),
        "critical_rank_shipped_fallback_seconds": float(critical["shipped"]),
        "critical_rank_fit_overhead_seconds": max(
            0.0, float(critical["fit"] - critical["adapted"]
                       - critical["shipped"])),
        "jobs_per_rank": [int(by_rank[rank]["jobs"])
                          for rank in range(world)],
        "job_count": len(rows),
        "refusal_count": sum(row["error"] is not None for row in rows),
    }


def _derived_pair_ceiling(specs, eta):
    """Bound the plan at ~2x its worst-case honest cost, derived not dialled.

    A crossing window needs about one node per ``eta`` of crossing bandwidth
    (``2A/eta`` for radius ``A``; see docs/theory/sigma-quadrature-problem.md
    section 7), and a sign-definite window a logarithmic count that is 8-12 in
    practice — 20 is a generous stand-in.  Doubling the sum leaves a healthy
    plan untouched while still catching a pathological fit that would otherwise
    run away.  This replaces the deck's ``mpa_sigma_max_nodes``: a user should
    not have to guess a resource ceiling the supports already determine.
    """
    total = 0.0
    for spec in specs:
        if spec["kind"] == "crossing":
            d = np.asarray(spec["problem"].denominators)
            radius = float(np.max(np.abs(d.real)))
            width = float(np.min(np.abs(d.imag)))
            total += 2.0 * radius / max(width, float(eta))
        else:
            total += 20.0
    return max(int(np.ceil(2.0 * total)), 32)


def _plan_cache_fingerprint(specs, *, eta, target, safety, factor_cap,
                            pair_ceiling, grid_mode, lattice_bins):
    """Hash the measured numerical problems that determine fitted rules."""
    digest = hashlib.sha256()

    def add_array(value):
        array = np.asarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        count = min(int(array.size), 4096)
        indices = np.linspace(
            0, max(int(array.size) - 1, 0), count,
            dtype=np.int64) if count else np.empty(0, np.int64)
        sample = np.take(array, indices)
        # Collective reductions can differ below roundoff between otherwise
        # identical P=4 restarts.  Seven significant digits remain two orders
        # tighter than the smallest candidate tolerance while being invariant
        # to that reduction noise over measures with different unit scales.
        # Sampling keeps lookup bounded.  A fingerprint mismatch still takes
        # the complete live validation path below before a rule can execute.
        if np.issubdtype(sample.dtype, np.complexfloating):
            canonical = "\0".join(
                f"{value.real:.7g},{value.imag:.7g}" for value in sample)
        elif np.issubdtype(sample.dtype, np.floating):
            canonical = "\0".join(f"{value:.7g}" for value in sample)
        else:
            canonical = np.ascontiguousarray(sample).view(np.uint8)
        digest.update(canonical.encode() if isinstance(canonical, str)
                      else canonical)

    digest.update(f"delivered-plan-cache-v{_PLAN_CACHE_VERSION}".encode())
    digest.update(repr((float(eta), float(target), float(safety),
                        float(factor_cap), int(pair_ceiling), str(grid_mode),
                        int(lattice_bins))).encode())
    for spec in specs:
        branch = spec["branch"]
        digest.update(repr((
            spec["name"], spec["kind"], float(spec["pole_sign"]),
            int(spec["pole_interval"]), branch.tag, branch.space,
            bool(branch.neg_omega_half))).encode())
        add_array((*spec["state_interval"], *spec["pole_bounds"],
                   spec["E_ref_A"], spec["envelope"]))
        for key in ("pole_indices", "state_indices", "raw_state_energy"):
            add_array(spec[key])
        for problem in (spec["problem"], spec["validation"]):
            for value in (problem.frequencies, problem.internal_sums,
                          problem.cell_masses):
                add_array(value)
            digest.update(repr((float(problem.excluded_radius),
                                float(problem.normalization_floor),
                                bool(problem.zero_weight_sum))).encode())
    return digest.hexdigest()


def _load_plan_cache(path, fingerprint, n_specs):
    if path is None:
        return None
    try:
        with open(path, "rb") as stream:
            payload = pickle.load(stream)
    except FileNotFoundError:
        return None
    except (OSError, pickle.PickleError, EOFError) as exc:
        raise RuntimeError(
            f"could not read delivered-plan cache {path!r}: {exc}") from exc
    if (not isinstance(payload, dict)
            or payload.get("version") != _PLAN_CACHE_VERSION):
        return None
    if payload.get("kind", "fits") != "fits":
        return None
    fits = payload.get("fits")
    if not isinstance(fits, list) or len(fits) != int(n_specs):
        raise RuntimeError(
            f"delivered-plan cache {path!r} has an invalid fit census")
    return (fits, int(payload["free_pairs"]),
            float(payload["required_cost"]),
            int(payload["window_tau_pairs"]),
            payload.get("fingerprint") == fingerprint)


def _validate_cached_fits(specs, fits, *, eta, factor_cap, pair_ceiling,
                          pointwise_budget):
    """Re-certify cached nodes and weights on the live measured problems."""
    if len(fits) != len(specs):
        return None
    for spec, fit in zip(specs, fits):
        try:
            times = np.asarray(fit["times"], np.complex128)
            weights = np.asarray(fit["weights"], np.complex128)
            residual_target = float(fit["residual_target"])
            metrics = _rule_metrics(spec["validation"], times, weights)
            factor = _factor_growth(spec, times, eta)
        except (KeyError, TypeError, ValueError, FloatingPointError,
                np.linalg.LinAlgError):
            return None
        if (times.ndim != 1 or weights.shape != times.shape
                or not times.size or np.any(times == 0.0)
                or not np.all(np.isfinite(times))
                or not np.all(np.isfinite(weights)) or not _rule_accepted(
                    metrics, residual_target)
                or max(factor) > float(factor_cap)):
            return None
        required = max(
            metrics[0], metrics[1] * RUNTIME_NOISE_EPSILON
            / AMPLIFICATION_NOISE_SAFETY)
        fit.update(metrics=metrics, factor_growth=factor,
                   required_target=required,
                   absolute_cost=float(spec["envelope"] * required))
    pointwise_cost = _pointwise_rule_costs(
        specs, fits, np.asarray(pointwise_budget).size)
    if (sum(int(np.asarray(fit["times"]).size) for fit in fits)
            > int(pair_ceiling)
            or np.any(pointwise_cost > np.asarray(pointwise_budget))):
        return None
    return float(np.max(pointwise_cost, initial=0.0))


def _save_plan_cache(path, fingerprint, fits, free_pairs, required_cost,
                     window_tau_pairs):
    """Atomically publish one rank's fitted-rule receipt."""
    if path is None or process_rank() != 0:
        return
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    payload = {
        "version": _PLAN_CACHE_VERSION,
        "kind": "fits",
        "fingerprint": fingerprint,
        "fits": fits,
        "free_pairs": int(free_pairs),
        "required_cost": float(required_cost),
        "window_tau_pairs": int(window_tau_pairs),
    }
    try:
        with open(temporary, "wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def delivered_plan_request_fingerprint(branches, omega_grid_ry, *,
                                       fit_ledger, parameters):
    """Identify the stable upstream inputs of a complete delivered plan."""
    digest = hashlib.sha256()
    digest.update(b"delivered-complete-plan-v1")

    def add(value):
        if isinstance(value, Mapping):
            digest.update(b"{")
            for key in sorted(value, key=str):
                add(str(key))
                add(value[key])
            digest.update(b"}")
        elif isinstance(value, (tuple, list)):
            digest.update(b"[")
            for item in value:
                add(item)
            digest.update(b"]")
        elif isinstance(value, (np.ndarray, jax.Array)):
            array = (np.asarray(gather_to_host(value))
                     if isinstance(value, jax.Array) else np.asarray(value))
            array = np.ascontiguousarray(array)
            digest.update(array.dtype.str.encode())
            digest.update(repr(array.shape).encode())
            digest.update(array.view(np.uint8))
        else:
            digest.update(repr(value).encode())
            digest.update(b"\0")

    add(fit_ledger)
    add(parameters)
    add(np.asarray(omega_grid_ry, np.float64))
    for branch in branches:
        add((branch.tag, branch.space, bool(branch.neg_omega_half)))
        add(branch.E_A)
        add(branch.base_mask_A)
        add(branch.omega_abs)
        add(branch.omega_idx)
        add(branch.band_weight)
    return digest.hexdigest()


def load_complete_delivered_sigma_plan(path, request_fingerprint, branches):
    """Load a complete certified product-window receipt before its census."""
    if path is None:
        return None
    try:
        with open(path, "rb") as stream:
            payload = pickle.load(stream)
    except FileNotFoundError:
        return None
    except (OSError, pickle.PickleError, EOFError) as exc:
        raise RuntimeError(
            f"could not read delivered-plan cache {path!r}: {exc}") from exc
    if (not isinstance(payload, dict)
            or payload.get("version") != _PLAN_CACHE_VERSION
            or payload.get("kind") != "complete"
            or payload.get("request_fingerprint") != request_fingerprint):
        return None
    rows = []
    for saved in payload.get("rows", ()):
        branch_index = int(saved["branch_index"])
        if not 0 <= branch_index < len(branches):
            raise RuntimeError(
                f"delivered-plan cache {path!r} has an invalid branch index")
        branch = branches[branch_index]
        window_data = dict(saved["window"])
        window_data["nodes"] = MinimaxNodes(
            t=jnp.asarray(saved["t"], dtype=jnp.complex128),
            alpha=jnp.asarray(saved["alpha"], dtype=jnp.complex128))
        window_data["mask_A"] = np.asarray(window_data["mask_A"], bool)
        window = _SigmaWindow(**window_data)
        rows.append(SharedSigmaWindow(
            window=window, E_A=branch.E_A,
            omega_abs=np.asarray(saved["omega_abs"], np.float64),
            omega_idx=np.asarray(saved["omega_idx"], np.int64),
            pole_indices=np.asarray(saved["pole_indices"], np.int32),
            bounds=np.asarray(saved["bounds"], np.float64),
            phase_real=np.asarray(saved["phase_real"], bool),
            band_weight=branch.band_weight))
    geometry = dict(payload["geometry"])
    geometry.update(plan_cache_status="complete_hit",
                    plan_cache_path=path, plan_seconds=0.0)
    return rows, geometry


def _save_complete_delivered_sigma_plan(path, request_fingerprint, output,
                                        specs, geometry, branches):
    """Atomically publish the fully constructed runtime-window receipt."""
    if path is None or request_fingerprint is None or process_rank() != 0:
        return
    branch_index = {id(branch): index for index, branch in enumerate(branches)}
    rows = []
    for row, spec in zip(output, specs):
        win = row.window
        rows.append({
            "branch_index": branch_index[id(spec["branch"])],
            "t": np.asarray(jax.device_get(win.nodes.t), np.complex128),
            "alpha": np.asarray(
                jax.device_get(win.nodes.alpha), np.complex128),
            "window": {
                "name": win.name,
                "mask_A": np.asarray(jax.device_get(win.mask_A), bool),
                "E_ref_A": win.E_ref_A, "E_ref_B": win.E_ref_B,
                "omega_sign": win.omega_sign, "project": win.project,
                "prefactor": win.prefactor,
                "mask_B_mode": win.mask_B_mode,
                "mask_B_threshold": win.mask_B_threshold,
                "crossing_kind": win.crossing_kind,
                "max_error": win.max_error, "provenance": win.provenance,
                "E_min": win.E_min, "E_max": win.E_max,
                "B_lo": win.B_lo, "B_hi": win.B_hi,
                "omega_indices": win.omega_indices,
            },
            "omega_abs": np.asarray(row.omega_abs, np.float64),
            "omega_idx": np.asarray(row.omega_idx, np.int64),
            "pole_indices": np.asarray(row.pole_indices, np.int32),
            "bounds": np.asarray(row.bounds, np.float64),
            "phase_real": np.asarray(row.phase_real, bool),
        })
    payload = {
        "version": _PLAN_CACHE_VERSION, "kind": "complete",
        "request_fingerprint": request_fingerprint,
        "rows": rows, "geometry": geometry,
    }
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    try:
        with open(temporary, "wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _per_branch(values, branches, name):
    if isinstance(values, Mapping):
        try:
            return [values[branch.tag] for branch in branches]
        except KeyError as exc:
            raise ValueError(
                f"{name} has no entry for branch {exc.args[0]!r}") from exc
    if isinstance(values, (np.ndarray, jax.Array)):
        raise ValueError(
            f"{name} must contain one array per branch; wrap a shared array "
            "once for each branch explicitly")
    rows = list(values)
    if len(rows) != len(branches):
        raise ValueError(
            f"{name} has {len(rows)} entries for {len(branches)} branches")
    return rows


def _optional_per_branch(values, branches, name):
    return ([None] * len(branches) if values is None
            else _per_branch(values, branches, name))


def _branch_states(branch, amplitude):
    """Return signed energies, delivered masses, and flat live indices."""
    energy = np.asarray(gather_to_host(branch.E_A), dtype=np.float64)
    mask = np.asarray(gather_to_host(branch.base_mask_A), dtype=bool)
    if energy.shape != mask.shape:
        mask = np.reshape(mask, energy.shape)
    if branch.band_weight is None:
        occupation = np.ones(energy.shape, dtype=np.float64)
    else:
        occupation = np.abs(np.asarray(
            gather_to_host(branch.band_weight), dtype=np.float64
        ).reshape(energy.shape))
    if amplitude is None:
        state_amplitude = np.ones(energy.shape, dtype=np.float64)
    else:
        state_amplitude = np.abs(np.asarray(
            gather_to_host(amplitude), dtype=np.complex128
        ).reshape(energy.shape))
    state_mass = occupation * state_amplitude
    live = (mask & np.isfinite(energy) & np.isfinite(state_mass)
            & (state_mass > 0.0))
    if not np.any(live):
        raise ValueError(
            f"delivered Sigma branch {branch.tag!r} has no live states")
    pole_sign = 1.0 if branch.space == "cond" else -1.0
    flat_live = np.flatnonzero(live.reshape(-1)).astype(np.int32)
    return (pole_sign * energy.reshape(-1)[flat_live],
            state_mass.reshape(-1)[flat_live], flat_live)


def delivered_product_geometry(branches, regularization_width_ry, *,
                               edge_factor=1.5):
    """Return the shared state/pole edges of the Cartesian construction."""
    branch_rows = list(branches)
    eta = float(regularization_width_ry)
    edge = float(edge_factor)
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("delivered Sigma regularization must be positive")
    if not np.isfinite(edge) or edge < 0.0:
        raise ValueError("edge_factor must be finite and nonnegative")
    omega_max = max((float(np.max(branch.omega_abs))
                     for branch in branch_rows if branch.omega_abs.size),
                    default=0.0)
    excursion = 0.0
    for branch in branch_rows:
        signed, _mass, _indices = _branch_states(branch, None)
        pole_sign = 1.0 if branch.space == "cond" else -1.0
        raw = pole_sign * signed
        excursion = max(excursion, -min(float(np.min(raw)), 0.0))
    state_edge = edge * eta
    pole_edge = omega_max + state_edge + excursion
    return {
        "omega_max_ry": omega_max,
        "state_edge_ry": state_edge,
        "pole_edge_ry": pole_edge,
        "negative_state_excursion_ry": excursion,
        "edge_factor": edge,
    }


def _leading_indices(index, count):
    first = index[0]
    if isinstance(first, slice):
        start, stop, step = first.indices(int(count))
        return np.arange(start, stop, step, dtype=np.int64)
    return np.asarray(first, dtype=np.int64).reshape(-1)


def _local_pole_chunks(Omega, B):
    """Yield host views of each unique local pole shard.

    This is the NumPy and single-process fallback. Distributed JAX arrays use
    :func:`measure_delivered_sigma_pole_fields`, which reduces their resident
    shards on the device and transfers only the bounded moment table.
    """
    if tuple(Omega.shape) != tuple(B.shape) or len(Omega.shape) < 1:
        raise ValueError("per-branch pole and residue arrays must match")
    if isinstance(Omega, jax.Array) != isinstance(B, jax.Array):
        raise ValueError("pole and residue arrays must use the same storage type")
    n_poles = int(Omega.shape[0])
    if not isinstance(Omega, jax.Array):
        if process_rank() == 0:
            yield np.arange(n_poles), np.asarray(Omega), np.asarray(B)
        return
    if bool(getattr(Omega, "is_fully_replicated", False)):
        if process_rank() == 0:
            yield (np.arange(n_poles), np.asarray(Omega.addressable_data(0)),
                   np.asarray(B.addressable_data(0)))
        return
    for shard_O, shard_B in zip(Omega.addressable_shards,
                                B.addressable_shards):
        if shard_O.index != shard_B.index:
            raise ValueError("pole and residue shard layouts differ")
        yield (_leading_indices(shard_O.index, n_poles),
               np.asarray(shard_O.data), np.asarray(shard_B.data))


def _axis_cloud_weights(values, nodes):
    """Return the two linear-interpolation cells and weights on one axis."""
    if nodes.size == 1:
        zero = np.zeros(values.size, dtype=np.int64)
        return ((zero, np.ones(values.size)),)
    lower = np.clip(np.searchsorted(nodes, values, side="right") - 1,
                    0, nodes.size - 2)
    width = nodes[lower + 1] - nodes[lower]
    fraction = np.where(
        width > 0.0,
        (values - nodes[lower]) / np.where(width > 0.0, width, 1.0),
        0.0)
    return ((lower, 1.0 - fraction), (lower + 1, fraction))


def _bounded_pole_moments(values, masses, bins, eta):
    """Reduce one host pole shard to a bounded two-dimensional lattice."""
    value = np.asarray(values, dtype=np.complex128).reshape(-1)
    mass = np.asarray(masses, dtype=np.float64).reshape(-1)
    bins = int(bins)
    if value.shape != mass.shape:
        raise ValueError("pole values and masses must have matching shapes")
    if bins < 4:
        raise ValueError("lattice_bins must be at least 4")
    if not value.size:
        return np.zeros((3, bins * bins), dtype=np.float64)
    intrinsic_width = np.maximum(-value.imag - float(eta), 0.0)
    real_coordinate = value.real / (value.real + float(eta))
    width_coordinate = intrinsic_width / (intrinsic_width + float(eta))
    nodes = np.linspace(0.0, 1.0, bins)
    moments = np.zeros((3, bins * bins), dtype=np.float64)
    for real_index, real_weight in _axis_cloud_weights(real_coordinate, nodes):
        for width_index, width_weight in _axis_cloud_weights(
                width_coordinate, nodes):
            index = real_index * bins + width_index
            share = mass * real_weight * width_weight
            np.add.at(moments[0], index, share)
            np.add.at(moments[1], index, share * value.real)
            np.add.at(moments[2], index, share * value.imag)
    return moments


def _sum_fixed_process_table(local, mesh_xy, label):
    if process_count() > 1 and mesh_xy is None:
        raise ValueError(
            f"distributed delivered planning needs mesh_xy to all-reduce "
            f"its bounded {label}")
    return psum_replicate(local, mesh_xy)


_DEVICE_POLE_REDUCERS = {}
_LAST_POLE_FIELD_MEASURE = None
# Measured on the Na 24-band census: one 49-million-value scatter serialized
# on 3,750 counters. Independent 4K-value tables keep about three values per
# counter and the temporary below 400 MB per pole; the collective stays 30 KB.
_CENSUS_HISTOGRAM_BLOCK = 4096


def _device_pole_reducer(Omega_local, B_local, bins, eta, split):
    """Return one cached shard-local reducer shared by every pole."""
    key = (tuple(Omega_local.shape), tuple(B_local.shape), int(bins),
           np.dtype(Omega_local.dtype).str, np.dtype(B_local.dtype).str,
           float(eta), float(split))
    cached = _DEVICE_POLE_REDUCERS.get(key)
    if cached is not None:
        return cached

    bins = int(bins)
    eta = float(eta)
    split = float(split)
    n_cells = bins * bins

    def _local_reduce(omega_local, residue_local, pole):
        """Reduce one local spatial tile to its 30 KB partial table."""
        omega = jnp.reshape(omega_local[pole], (-1,))
        residue = jnp.reshape(residue_local[pole], (-1,))
        finite_residue = jnp.isfinite(residue)
        residue_live = finite_residue & (jnp.abs(residue) > 0.0)
        gamma = -jnp.imag(omega)
        pole_ok = (jnp.isfinite(omega) & (jnp.real(omega) > 0.0)
                   & (gamma >= 0.0))
        live = residue_live & pole_ok

        real = jnp.where(live, jnp.real(omega), eta)
        width = jnp.where(live, gamma, 0.0)
        mass = jnp.where(live, jnp.abs(residue), 0.0)
        imag = -(width + eta)
        real_coordinate = real / (real + eta)
        width_coordinate = width / (width + eta)

        real_scaled = real_coordinate * (bins - 1)
        width_scaled = width_coordinate * (bins - 1)
        real_lower = jnp.clip(
            jnp.floor(real_scaled), 0, bins - 2).astype(jnp.int32)
        width_lower = jnp.clip(
            jnp.floor(width_scaled), 0, bins - 2).astype(jnp.int32)
        real_fraction = real_scaled - real_lower
        width_fraction = width_scaled - width_lower
        interval = (real > split).astype(jnp.int32)
        block_size = _CENSUS_HISTOGRAM_BLOCK
        n_values = int(omega.size)
        n_blocks = (n_values + block_size - 1) // block_size
        pad = n_blocks * block_size - n_values

        def _blocks(value, fill):
            return jnp.pad(value, (0, pad), constant_values=fill).reshape(
                n_blocks, block_size)

        block_inputs = (
            _blocks(interval, 0),
            _blocks(real_lower, 0),
            _blocks(width_lower, 0),
            _blocks(real_fraction, 0.0),
            _blocks(width_fraction, 0.0),
            _blocks(mass, 0.0),
            _blocks(real, eta),
            _blocks(imag, -eta),
        )

        def _block_moments(block_interval, block_real_lower,
                           block_width_lower, block_real_fraction,
                           block_width_fraction, block_mass, block_real,
                           block_imag):
            components = jnp.arange(3, dtype=jnp.int32)[:, None]

            def _add_corner(corner, block_moment):
                real_upper = corner // 2
                width_upper = corner % 2
                real_weight = jnp.where(
                    real_upper == 1, block_real_fraction,
                    1.0 - block_real_fraction)
                width_weight = jnp.where(
                    width_upper == 1, block_width_fraction,
                    1.0 - block_width_fraction)
                cell = ((block_real_lower + real_upper) * bins
                        + block_width_lower + width_upper)
                share = block_mass * real_weight * width_weight
                values = jnp.stack(
                    (share, share * block_real, share * block_imag), axis=0)
                return block_moment.at[
                    block_interval[None, :], components, cell[None, :]
                ].add(values)

            return jax.lax.fori_loop(
                0, 4, _add_corner,
                jnp.zeros((2, 3, n_cells), dtype=jnp.float64))

        moments = jnp.sum(jax.vmap(_block_moments)(*block_inputs), axis=0)
        counts = jnp.stack((
            jnp.count_nonzero(~finite_residue),
            jnp.count_nonzero(residue_live & ~pole_ok),
            jnp.count_nonzero(live),
        )).astype(jnp.int64)
        return jnp.concatenate(
            (jnp.reshape(moments, (-1,)), counts.astype(jnp.float64)))

    cached = jax.jit(_local_reduce)
    _DEVICE_POLE_REDUCERS[key] = cached
    return cached


def _host_pole_moments(Omega, B, eta, bins, split, mesh_xy):
    """Build the bounded pole table with NumPy for small host inputs."""
    n_poles = int(Omega.shape[0])
    local_moments = np.zeros(
        (n_poles, 2, 3, int(bins) ** 2), dtype=np.float64)
    bad_B = bad_pole = live_count = 0
    for pole_indices, Omega_chunk, B_chunk in _local_pole_chunks(Omega, B):
        for local, pole_index in enumerate(pole_indices):
            omega = np.asarray(Omega_chunk[local], np.complex128).reshape(-1)
            residue = np.asarray(B_chunk[local], np.complex128).reshape(-1)
            finite_B = np.isfinite(residue)
            bad_B += int(np.count_nonzero(~finite_B))
            live = finite_B & (np.abs(residue) > 0.0)
            gamma = -omega.imag
            finite_O = np.isfinite(omega)
            bad_pole += int(np.count_nonzero(
                live & (~finite_O | (omega.real <= 0.0) | (gamma < 0.0))))
            live &= finite_O & (omega.real > 0.0) & (gamma >= 0.0)
            if not np.any(live):
                continue
            broadened = omega[live].real - 1.0j * (gamma[live] + eta)
            residue_mass = np.abs(residue[live])
            shallow = broadened.real <= split
            for interval, selected in enumerate((shallow, ~shallow)):
                if np.any(selected):
                    local_moments[int(pole_index), interval] += (
                        _bounded_pole_moments(
                            broadened[selected], residue_mass[selected],
                            bins, eta))
            live_count += int(np.count_nonzero(live))

    bad = _sum_fixed_process_table(
        np.asarray([bad_B, bad_pole], dtype=np.int64), mesh_xy,
        "refusal-count table")
    moments = np.empty_like(local_moments)
    for pole in range(n_poles):
        for interval in range(2):
            moments[pole, interval] = _sum_fixed_process_table(
                local_moments[pole, interval], mesh_xy,
                "pole-interval mass/moment lattice")
    live_global = int(_sum_fixed_process_table(
        np.asarray(live_count, dtype=np.int64), mesh_xy,
        "live-pole-count scalar"))
    return moments, np.asarray(bad, np.int64), live_global


def _pole_fields_from_moments(moments, bad, live_count, bins, split,
                              pole_offset, *, reduction):
    """Convert a global bounded moment table into compact pole cells."""
    if int(bad[0]):
        raise ValueError(
            f"delivered Sigma poles contain {int(bad[0])} nonfinite residues")
    if int(bad[1]):
        raise ValueError(
            f"delivered Sigma poles contain {int(bad[1])} unsupported live poles")

    n_poles = int(moments.shape[0])
    pole_cells, pole_weights = [], []
    for pole_index in range(n_poles):
        cells_by_interval, weights_by_interval = [], []
        for interval in range(2):
            row = moments[pole_index, interval]
            live = row[0] > 0.0
            if not np.any(live):
                cells_by_interval.append(None)
                weights_by_interval.append(None)
                continue
            weights = row[0, live]
            cells_by_interval.append(
                (row[1, live] + 1.0j * row[2, live]) / weights)
            weights_by_interval.append(weights)
        pole_cells.append(tuple(cells_by_interval))
        pole_weights.append(tuple(weights_by_interval))

    ceiling = int(bins) ** 2
    evidence = {
        "pole_split_ry": split,
        "local_spatial_cell_ceiling_per_pole_interval": ceiling,
        "collective_spatial_cell_ceiling_per_pole_interval": ceiling,
        "collective_payload_bytes_per_pole_per_rank": (
            2 * 3 * ceiling * np.dtype(np.float64).itemsize),
        "collective_reduction": reduction,
        "collective_ceiling_independent_of_process_count": True,
        "collective_ceiling_independent_of_state_count": True,
        "collective_ceiling_independent_of_spatial_extent": True,
    }
    poles = np.arange(int(pole_offset), int(pole_offset) + n_poles,
                      dtype=np.int32)
    return (tuple(pole_cells), tuple(pole_weights), poles, int(live_count),
            evidence)


def measure_delivered_sigma_pole_fields(
    Omega, B, *, regularization_width_ry, pole_split_ry,
    lattice_bins=DEFAULT_LATTICE_BINS, pole_offset=0, mesh_xy=None,
):
    """Reduce one pole batch on its resident shards.

    The pole locations and residue masses do not depend on the causal state
    branch. Distributed JAX inputs stay on their device shards while one
    kernel checks every pole and builds a fixed mass/first-moment lattice.
    One tree transfer copies the batch's small tables to the host. The process
    collective sums only those fixed 30 KB-per-pole tables. NumPy inputs keep
    a simple host fallback for tests.

    Returns
    -------
    tuple
        Compact cells, masses, global pole indices, live spatial-pole count,
        and bounded-reduction evidence.
    """
    if tuple(Omega.shape) != tuple(B.shape) or len(Omega.shape) < 1:
        raise ValueError("per-branch pole and residue arrays must match")
    if isinstance(Omega, jax.Array) != isinstance(B, jax.Array):
        raise ValueError("pole and residue arrays must use the same storage type")
    eta = float(regularization_width_ry)
    split = float(pole_split_ry)
    bins = int(lattice_bins)
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("regularization_width_ry must be finite and positive")
    if not np.isfinite(split) or split <= 0.0:
        raise ValueError("pole_split_ry must be finite and positive")
    if bins < 4:
        raise ValueError("lattice_bins must be at least 4")

    if isinstance(Omega, jax.Array) and mesh_xy is not None:
        started = time.perf_counter()
        if getattr(Omega.sharding, "spec", None) != getattr(
                B.sharding, "spec", None):
            raise ValueError("pole and residue shard layouts differ")
        omega_local = Omega.addressable_data(0)
        residue_local = B.addressable_data(0)
        factory_started = time.perf_counter()
        reducer = _device_pole_reducer(
            omega_local, residue_local, bins, eta, split)
        factory_seconds = time.perf_counter() - factory_started
        submit_seconds = []
        local_payload_rows = []
        for pole in range(int(Omega.shape[0])):
            submit_started = time.perf_counter()
            local_payload_rows.append(reducer(
                omega_local, residue_local, jnp.asarray(pole, jnp.int32))
            )
            submit_seconds.append(time.perf_counter() - submit_started)
        readback_started = time.perf_counter()
        local_payload = np.asarray(
            jax.device_get(tuple(local_payload_rows)), np.float64)
        readback_seconds = time.perf_counter() - readback_started
        collective_started = time.perf_counter()
        payload = np.asarray(_sum_fixed_process_table(
            local_payload, mesh_xy, "pole-batch mass/moment table"),
            np.float64)
        collective_seconds = time.perf_counter() - collective_started
        n_moments = 2 * 3 * bins * bins
        moments = payload[:, :n_moments].reshape(
            int(Omega.shape[0]), 2, 3, bins * bins)
        counts = np.sum(
            np.rint(payload[:, n_moments:]).astype(np.int64), axis=0)
        reduction = "device_local_fixed_mass_first_moment_psum"
        if os.environ.get("LORRAX_DELIVERED_CENSUS_PROFILE", "0") == "1":
            print(
                f"[delivered-census-profile] rank={process_rank()} "
                f"device_field_measure={time.perf_counter() - started:.6f}s "
                f"poles={int(Omega.shape[0])} host_bytes={payload.nbytes} "
                f"factory={factory_seconds:.6f}s "
                f"submit={','.join(f'{value:.6f}' for value in submit_seconds)}s "
                f"readback={readback_seconds:.6f}s "
                f"collective={collective_seconds:.6f}s",
                flush=True)
    else:
        moments, bad, live_count = _host_pole_moments(
            Omega, B, eta, bins, split, mesh_xy)
        counts = np.asarray((bad[0], bad[1], live_count), np.int64)
        reduction = "two_fixed_mass_first_moment_psums"
    return _pole_fields_from_moments(
        moments, counts[:2], counts[2], bins, split, pole_offset,
        reduction=reduction)


def _cached_pole_field_measure(Omega, B, **parameters):
    """Reuse the last JAX field table while branches share one batch."""
    global _LAST_POLE_FIELD_MEASURE
    if isinstance(Omega, jax.Array):
        key = tuple(sorted(parameters.items(), key=lambda item: item[0]))
        cached = _LAST_POLE_FIELD_MEASURE
        if (cached is not None and cached[0]() is Omega
                and cached[1]() is B and cached[2] == key):
            return cached[3]
        measured = measure_delivered_sigma_pole_fields(
            Omega, B, **parameters)
        _LAST_POLE_FIELD_MEASURE = (
            weakref.ref(Omega), weakref.ref(B), key, measured)
        return measured
    return measure_delivered_sigma_pole_fields(Omega, B, **parameters)


def _pole_measures(branch, Omega, B, eta, amplitude, bins, pole_split_ry,
                   *, pole_offset=0, mesh_xy=None):
    """Measure a pole batch and attach one branch's small state table."""
    return measure_delivered_sigma_pole_batch(
        branch, Omega, B, regularization_width_ry=eta,
        pole_split_ry=pole_split_ry, state_amplitude=amplitude,
        lattice_bins=bins, pole_offset=pole_offset, mesh_xy=mesh_xy)


def measure_delivered_sigma_pole_batch(
    branch, Omega, B, *, regularization_width_ry, pole_split_ry=None,
    state_amplitude=None, lattice_bins=DEFAULT_LATTICE_BINS, pole_offset=0,
    mesh_xy=None, pole_field_measure=None,
):
    """Attach one causal branch to a shared compact pole-field measure.

    Pole cells and masses do not depend on the state branch. Pass a prior
    ``pole_field_measure`` to attach another branch without reducing the large
    pole field again.
    """
    if pole_split_ry is None:
        pole_split_ry = delivered_product_geometry(
            [branch], regularization_width_ry)["pole_edge_ry"]
    parameters = dict(
        regularization_width_ry=float(regularization_width_ry),
        pole_split_ry=float(pole_split_ry),
        lattice_bins=int(lattice_bins), pole_offset=int(pole_offset),
        mesh_xy=mesh_xy)
    if pole_field_measure is None:
        pole_field_measure = _cached_pole_field_measure(
            Omega, B, **parameters)
    pole_cells, pole_weights, poles, live_count, evidence = pole_field_measure
    signed_energy, state_mass, state_indices = _branch_states(
        branch, state_amplitude)
    raw_count = int(live_count) * int(signed_energy.size)
    return (signed_energy, state_mass, state_indices, pole_cells, pole_weights,
            poles, raw_count, evidence)


def combine_delivered_sigma_pole_measures(batch_measures):
    """Combine consecutive bounded pole batches in leading-pole order."""
    rows = list(batch_measures)
    if not rows:
        raise ValueError("delivered Sigma needs at least one pole batch")
    signed = np.asarray(rows[0][0], np.float64)
    state_mass = np.asarray(rows[0][1], np.float64)
    state_indices = np.asarray(rows[0][2], np.int32)
    evidence = dict(rows[0][7])
    cells, weights, poles, raw_count = [], [], [], 0
    for row in rows:
        if (not np.array_equal(row[0], signed)
                or not np.array_equal(row[1], state_mass)
                or not np.array_equal(row[2], state_indices)):
            raise ValueError("delivered pole batches disagree about states")
        if dict(row[7]) != evidence:
            raise ValueError("delivered pole batches disagree about geometry")
        cells.extend(row[3])
        weights.extend(row[4])
        poles.extend(np.asarray(row[5], np.int32).tolist())
        raw_count += int(row[6])
    order = np.argsort(np.asarray(poles), kind="stable")
    return (signed, state_mass, state_indices,
            tuple(cells[index] for index in order),
            tuple(weights[index] for index in order),
            np.asarray(poles, np.int32)[order], raw_count, evidence)


def _product_problem(state_positions, pole_bounds, measure, frequencies,
                     pole_sign, bins):
    signed, state_mass, _state_indices, pole_cells, pole_weights, poles, *_ = measure
    state_positions = np.asarray(state_positions, np.int64)
    cells, masses, selected_poles = [], [], []
    pole_lo, pole_hi = map(float, pole_bounds)
    for local, pole in enumerate(np.asarray(poles, np.int32)):
        pole_selected = False
        for part in (0, 1):
            pole_cell = pole_cells[local][part]
            pole_weight = pole_weights[local][part]
            if pole_cell is None:
                continue
            pole_cell = np.asarray(pole_cell)
            keep = ((pole_cell.real > pole_lo)
                    & (pole_cell.real <= pole_hi))
            if not np.any(keep):
                continue
            pole_selected = True
            internal = (np.asarray(signed)[state_positions, None]
                        + float(pole_sign) * pole_cell[None, keep])
            mass = (np.asarray(state_mass)[state_positions, None]
                    * np.asarray(pole_weight)[None, keep])
            cells.append(internal.reshape(-1))
            masses.append(mass.reshape(-1))
        if pole_selected:
            selected_poles.append(int(pole))
    if not cells:
        return None
    internal = np.concatenate(cells)
    delivered = np.concatenate(masses)
    base_cells, base_mass, refined_cells, refined_mass = (
        tail_refined_lattice_measure(
            internal, delivered, bins_per_axis=int(bins)))
    return (
        ReciprocalMeasureProblem(
            frequencies=frequencies, internal_sums=base_cells,
            cell_masses=base_mass),
        ReciprocalMeasureProblem(
            frequencies=frequencies, internal_sums=refined_cells,
            cell_masses=refined_mass),
        np.asarray(selected_poles, dtype=np.int32),
    )


def _window_kind(problem):
    real = problem.denominators.real
    if np.all(real > 0.0):
        return "sign_definite_positive"
    if np.all(real < 0.0):
        return "sign_definite_negative"
    return "crossing"


def _rule_metrics(problem, times, weights):
    """Measure residual and amplification from one shared phase matrix."""
    kept, delivered, _excluded = problem.retained()
    denominator = problem.denominators
    phase = np.exp(
        1.0j * denominator[..., None]
        * np.asarray(times, np.complex128)[None, None, :])
    term = phase * np.asarray(weights, np.complex128)[None, None, :]
    value = np.sum(term, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        truth = np.where(
            kept, 1.0 / np.where(kept, denominator, 1.0), 0.0)
    numerator = (np.where(kept, np.abs(value - truth), 0.0)
                 @ problem.cell_masses)
    residual = numerator / delivered

    kappa = (np.sum(np.abs(term), axis=-1)
             / np.maximum(np.abs(value), 1.0e-300))[kept]
    mass = np.broadcast_to(
        problem.cell_masses[None, :], denominator.shape)[kept]
    order = np.argsort(kappa, kind="stable")
    cumulative = np.cumsum(mass[order])
    p99 = kappa[order][
        np.searchsorted(cumulative, 0.99 * cumulative[-1])]
    return float(np.max(residual)), float(p99), float(np.max(kappa))


def _rule_accepted(metrics, target):
    """Apply the residual gate and the runtime-noise budget exactly."""
    return bool(
        metrics[0] <= float(target)
        and metrics[1] * RUNTIME_NOISE_EPSILON
        <= AMPLIFICATION_NOISE_SAFETY * float(target))


def _absolute_kernel_target(problem, relative_target):
    """Convert delivered relative error to a uniform absolute kernel bound."""
    _kept, delivered_mass, _excluded = problem.retained()
    total_mass = float(np.sum(problem.cell_masses))
    target = (float(relative_target) * float(np.min(delivered_mass))
              / total_mass)
    if not np.isfinite(target) or target <= 0.0:
        raise RuntimeError(
            "delivered product window has no positive absolute error target")
    return target


def _sign_definite_orientation(problem):
    """Return positive-real lower-half support and its executor transform."""
    denominator = problem.denominators
    if np.all(denominator.real > 0.0):
        if np.all(denominator.imag <= 0.0):
            return denominator, "positive_lower"
        if np.all(denominator.imag >= 0.0):
            return np.conj(denominator), "positive_upper"
    elif np.all(denominator.real < 0.0):
        if np.all(denominator.imag >= 0.0):
            return -denominator, "negative_upper"
        if np.all(denominator.imag <= 0.0):
            return -np.conj(denominator), "negative_lower"
    raise RuntimeError(
        "sign-definite product support crosses an axis and cannot be served")


def _sign_definite_candidates(problem, relative_target, max_nodes):
    """Yield on-demand noncrossing rules until the consumer accepts one."""
    from minimax import noncrossing_grids  # noqa: PLC0415

    rotated, transform = _sign_definite_orientation(problem)
    x_min = float(np.min(rotated.real))
    x_max = float(np.max(rotated.real))
    if not 0.0 < x_min <= x_max < np.inf:
        raise RuntimeError(
            f"invalid sign-definite support [{x_min:.6g}, {x_max:.6g}] Ry")
    range_value = x_max / x_min
    absolute_target = _absolute_kernel_target(problem, relative_target)
    scaled_target = absolute_target * x_min
    next_rank = 2
    while next_rank <= int(max_nodes):
        tau, alpha, rank, achieved = noncrossing_grids(
            range_value, scaled_target, N_start=next_rank,
            N_max=int(max_nodes) if next_rank == 2 else next_rank)
        tau = np.asarray(tau, np.float64) / x_min
        alpha = np.asarray(alpha, np.float64) / x_min
        if transform.startswith("positive"):
            times, weights = 1.0j * tau, alpha
        else:
            times, weights = -1.0j * tau, -alpha
        yield np.asarray(times), np.asarray(weights), {
            "family": "noncrossing_on_demand",
            "transform": transform,
            "requested_range": range_value,
            "fit_range": float(range_value),
            "requested_scaled_error": scaled_target,
            "fit_achieved_abs_error": float(achieved / x_min),
            "candidate_tolerance": float(scaled_target),
            "provenance": (
                f"on-demand Hackbusch-seeded noncrossing fit, R "
                f"{range_value:.6g}, rank {int(rank)}"),
        }
        # A real-axis fit already at roundoff carries no information about
        # an off-axis miss.  More ranks solve the same degenerate real problem,
        # so hand that support to the general measure-adapted fallback.
        if (int(rank) >= int(max_nodes)
                or float(achieved) <= 16.0 * np.finfo(np.float64).eps):
            return
        next_rank = int(rank) + 1


def _crossing_omega_patches(problem, measure, state_positions, pole_bounds,
                            pole_sign, state_edge, bins):
    """Return one measured window; on-demand fitting has no catalog span."""
    del measure, state_positions, pole_sign, state_edge, bins
    omega_rows = np.arange(problem.frequencies.size, dtype=np.int64)
    return ((omega_rows, (("identity", tuple(map(float, pole_bounds))),)),)


def _rule_candidate(problem, validation, times, weights, evidence):
    times = np.asarray(times, np.complex128)
    weights = np.asarray(weights, np.complex128)
    if (times.ndim != 1 or weights.shape != times.shape or not times.size
            or np.any(times == 0.0)
            or not np.all(np.isfinite(times))
            or not np.all(np.isfinite(weights))):
        raise RuntimeError("served quadrature has invalid or zero time nodes")
    return {
        "times": times,
        "weights": weights,
        "fit_metrics": _rule_metrics(problem, times, weights),
        "metrics": _rule_metrics(validation, times, weights),
        "evidence": evidence,
    }


def _selected_pole_values(spec):
    """Return the measured pole cells consumed by one product window."""
    cells = []
    pole_lo, pole_hi = map(float, spec["pole_bounds"])
    measure = spec["measure"]
    selected_poles = set(np.asarray(spec["pole_indices"]).tolist())
    for local, pole in enumerate(np.asarray(measure[5], np.int32)):
        if pole not in selected_poles:
            continue
        for part in (0, 1):
            pole_cells = measure[3][local][part]
            if pole_cells is not None:
                pole_cells = np.asarray(pole_cells)
                keep = ((pole_cells.real > pole_lo)
                        & (pole_cells.real <= pole_hi))
                if np.any(keep):
                    cells.append(pole_cells[keep])
    if not cells:
        raise RuntimeError(
            f"delivered product window {spec['name']!r} has no pole cells")
    return np.concatenate(cells)


def _factor_references(spec):
    """Choose exact executor shifts that keep imaginary-time factors bounded."""
    raw = np.asarray(spec["raw_state_energy"], dtype=np.float64)
    if spec["kind"] == "crossing":
        return float(np.min(raw)), 0.0
    _rotated, transform = _sign_definite_orientation(spec["problem"])
    table_sign = 1.0 if transform.startswith("positive") else -1.0
    use_upper = float(spec["pole_sign"]) * table_sign > 0.0
    endpoint = np.max if use_upper else np.min
    poles = _selected_pole_values(spec)
    return float(endpoint(raw)), float(endpoint(poles.real))


def _factor_growth(spec, times, eta):
    time = np.asarray(times, dtype=np.complex128).reshape(-1)
    if not time.size:
        return 0.0, 0.0
    pole_sign = float(spec["pole_sign"])
    time_exec = pole_sign * time
    raw = np.asarray(spec["raw_state_energy"], dtype=np.float64)
    reference = float(spec["E_ref_A"])
    green = float(np.max(np.real(
        -1.0j * (raw[:, None] - reference) * time_exec[None, :])))
    pole_values = _selected_pole_values(spec) + 1.0j * eta
    pole_reference = float(spec.get("E_ref_B", 0.0))
    screened = float(np.max(np.real(
        -1.0j * (pole_values[:, None] - pole_reference)
        * time_exec[None, :])))
    return green, screened


def _measure_adapted_candidate(spec, eta, max_nodes, factor_growth_cap,
                               relative_target):
    """Measure-adapted rule for one product window, or ``None`` on refusal.

    Node count is the runtime currency: one node is one FFT(G_w W_w).  The
    The contour, rank, and nodes are derived from this run's measure.  This is
    the production path for crossing supports and the accuracy-preserving
    fallback for sign-definite supports that miss their noncrossing fit gates.
    """
    from minimax import (RoqPlanningRefusal, RoqWindow,
                         plan_measure_adapted_roq)  # noqa: PLC0415

    d = spec["problem"].denominators
    sigma = 1 if float(np.mean(np.imag(d) > 0.0)) > 0.5 else -1
    window = RoqWindow(
        name=spec["name"], fit=spec["problem"], validation=spec["validation"],
        target=float(relative_target), branch=spec["name"], sigma=int(sigma))
    try:
        plan = plan_measure_adapted_roq((window,), float(eta))
    except RoqPlanningRefusal as exc:
        raise _ProductWindowRefusal(
            f"delivered product window {spec['name']!r} refused: achieved "
            f"(residual={exc.residual:.6g}, "
            f"amplification_p99={exc.kappa_p99:.6g}); the measure-adapted "
            "fit did not survive the residual and noise gates",
            exc.residual, exc.kappa_p99) from exc
    except (ValueError, RuntimeError, FloatingPointError, OverflowError,
            np.linalg.LinAlgError):
        return None
    if len(plan.rules) != 1:
        return None
    rule = plan.rules[0]
    times = np.asarray(rule.times, np.complex128)
    weights = np.asarray(rule.weights, np.complex128)
    if not times.size or int(times.size) > int(max_nodes):
        return None
    metrics = _rule_metrics(spec["validation"], times, weights)
    factor = _factor_growth(spec, times, eta)
    if (not _rule_accepted(metrics, relative_target)
            or max(factor) > float(factor_growth_cap)):
        return None
    required = max(metrics[0], metrics[1] * RUNTIME_NOISE_EPSILON
                   / AMPLIFICATION_NOISE_SAFETY)
    return {
        "times": times, "weights": weights,
        "metrics": metrics, "fit_metrics": metrics,
        "required_target": float(required),
        "absolute_cost": float(spec["envelope"] * required),
        "factor_growth": factor, "attempts": [],
        "evidence": {
            "family": "measure_adapted_roq",
            "candidate_tolerance": float(relative_target),
            "provenance": (f"measure-adapted ROQ, contour "
                           f"{rule.angle_deg:.1f} deg, rank {rule.rank}"),
        },
    }


def _merge_branch_specs(group):
    """One product window covering a branch's windows, or None if they differ.

    Valid only because a branch's windows tile a rectangle: their union is
    (union of states) x (union of poles), which SharedSigmaWindow already
    expresses as an arbitrary state mask plus an arbitrary pole list.  Whether
    merging PAYS is a separate, measured question — it does on valence
    (67 -> 12 nodes) and does not on conduction, whose merged scale span is
    unbounded and whose rules then need kappa 1e5 and worse (claim 525).
    """
    first = group[0]
    for spec in group[1:]:
        if (spec["branch"] is not first["branch"]
                or not np.array_equal(spec["omega_idx"], first["omega_idx"])):
            return None
    positions = np.unique(np.concatenate(
        [np.asarray(spec["state_positions"]).reshape(-1) for spec in group]))
    order = {int(p): None for p in positions}
    raw = np.empty(positions.size, dtype=np.float64)
    for spec in group:
        for value, position in zip(np.asarray(spec["raw_state_energy"]),
                                   np.asarray(spec["state_positions"]).reshape(-1)):
            raw[np.searchsorted(positions, int(position))] = float(value)
    del order
    problem = ReciprocalMeasureProblem(
        first["problem"].frequencies,
        np.concatenate([spec["problem"].internal_sums for spec in group]),
        np.concatenate([spec["problem"].cell_masses for spec in group]))
    validation = ReciprocalMeasureProblem(
        first["validation"].frequencies,
        np.concatenate([spec["validation"].internal_sums for spec in group]),
        np.concatenate([spec["validation"].cell_masses for spec in group]))
    envelope_by_frequency = (
        problem.cell_masses[None, :] / np.abs(problem.denominators)).sum(axis=1)
    merged = dict(first)
    merged.update(
        name=f"{first['branch'].tag}:consolidated",
        problem=problem, validation=validation,
        kind=("crossing" if any(spec["kind"] == "crossing" for spec in group)
              else first["kind"]),
        pole_indices=np.unique(np.concatenate(
            [np.asarray(spec["pole_indices"]).reshape(-1) for spec in group])),
        state_positions=positions,
        state_indices=np.unique(np.concatenate(
            [np.asarray(spec["state_indices"]).reshape(-1) for spec in group])),
        raw_state_energy=raw,
        state_interval=(min(spec["state_interval"][0] for spec in group),
                        max(spec["state_interval"][1] for spec in group)),
        pole_bounds=(min(spec["pole_bounds"][0] for spec in group),
                     max(spec["pole_bounds"][1] for spec in group)),
        # pole_interval is the shallow/deep INDEX, not a bounds pair; a merged
        # window spans both, so -1 records "no single interval" for the
        # fingerprint and the report.  The real range lives in pole_bounds.
        pole_interval=-1,
        E_ref_A=float(np.min(raw)),
        envelope=float(np.max(envelope_by_frequency)),
        envelope_by_frequency=envelope_by_frequency)
    merged["E_ref_A"], merged["E_ref_B"] = _factor_references(merged)
    return merged


def _consolidate_branches(specs, candidates, eta, max_nodes, factor_cap,
                          pointwise_budget, trial_candidates=None):
    """Replace a branch's windows by one merged window when that costs less.

    Tried, not assumed: the merged window is fitted and kept only if it is
    accepted AND uses strictly fewer nodes than the windows it replaces.
    Fewer windows also means fewer spatial sweeps, so this cuts both currencies.
    """
    by_branch = {}
    for index, spec in enumerate(specs):
        by_branch.setdefault(id(spec["branch"]), []).append(index)
    trials = []
    for indices in by_branch.values():
        if len(indices) < 2:
            continue
        group = [specs[i] for i in indices]
        merged = _merge_branch_specs(group)
        if merged is None:
            continue
        allowance = _pointwise_window_allowance(merged, pointwise_budget)
        split_nodes = sum(int(candidates[i][0]["times"].size)
                          for i in indices)
        # A merged rule is only ever KEPT below the split's node count, so
        # searching at or above it can never yield an accepted rule.  Bounding
        # each trial by its own acceptance criterion is dial-free and removes
        # the dominant residual planning cost: a refused merged-conduction
        # trial measured 32.4 s of a 44.1 s plan searching ranks it could not
        # have kept.
        trial_ceiling = min(int(max_nodes), split_nodes - 1)
        if trial_ceiling < 1:
            continue
        trials.append((indices, merged, allowance, trial_ceiling))

    def fit_trial(index):
        _indices, merged, allowance, ceiling = trials[index]
        rows, detail = _window_candidates_profiled(
            merged, eta, ceiling, factor_cap, allowance,
            adapted_only=True)
        return rows[0], detail

    if trial_candidates is None:
        trial_candidates, trial_rows = _run_parallel_planner_jobs(
            len(trials), fit_trial, refuse_errors=False)
    else:
        if len(trial_candidates) != len(trials):
            raise RuntimeError(
                "cached consolidation trial count changed across retry")
        trial_rows = []
    keep = list(range(len(specs)))
    merged_specs, merged_candidates = dict(), dict()
    for (indices, merged, _allowance, _ceiling), candidate in zip(
            trials, trial_candidates):
        if candidate is None:
            continue
        split_nodes = sum(int(candidates[i][0]["times"].size) for i in indices)
        if int(candidate["times"].size) >= split_nodes:
            continue
        merged_specs[indices[0]] = merged
        merged_candidates[indices[0]] = [candidate]
        for dropped in indices[1:]:
            keep.remove(dropped)
    if not merged_specs:
        return specs, candidates, trial_rows, trial_candidates
    return ([merged_specs.get(i, specs[i]) for i in keep],
            [merged_candidates.get(i, candidates[i]) for i in keep],
            trial_rows, trial_candidates)


def window_candidates(spec, eta, max_nodes, factor_growth_cap,
                      relative_target, *, adapted_only=False):
    """Rules for one window: rotated ROQ or on-demand noncrossing fit.

    The single routing decision, at module level so the planner, the tests and
    the offline node audit all take the same path.
    """
    return _window_candidates_profiled(
        spec, eta, max_nodes, factor_growth_cap, relative_target,
        adapted_only=adapted_only)[0]


def _window_candidates_profiled(spec, eta, max_nodes, factor_growth_cap,
                                relative_target, *, adapted_only=False):
    """Return one window's rules and a cost split for planner evidence."""
    del adapted_only
    detail = {"adapted_fit_seconds": 0.0,
              "shipped_fallback_seconds": 0.0}
    started = time.perf_counter()
    try:
        candidates = _candidate_rules(
            spec, eta, max_nodes, factor_growth_cap, relative_target)
    finally:
        detail["adapted_fit_seconds"] = time.perf_counter() - started
    return candidates, detail


UNIFORM_RULE_TIME_BUDGET_SECONDS = 15.0   # Gauss reduction cap per window; the interpolatory rule is ready after ~1 s


def _uniform_rule_budget():
    """Seconds the Gauss reduction may spend per window.

    ``LORRAX_UNIFORM_RULE_BUDGET_S`` overrides the 15 s default.  The rule
    depends only on (box, eps), so the budget trades planning seconds for
    nodes: about 15 s keeps the crossing windows near 0.75 of the family
    rank, about 100 s reaches the Gauss count (~0.5 rank).  Measured on Na
    8x8x8: the -5..+5 eV crossing rules go 95 -> 45 nodes between 15 s and
    120 s at eps 1e-3 and the reduction stops by itself at ~57 s; the
    +-15 eV ones (rank ~350) are still improving at 120 s.  Windows are
    fitted one per rank (``_run_parallel_planner_jobs``), so the plan wall
    is ONE crossing window's budget plus geometry (~9 s) and exchange, not
    the sum over windows.  The right setting is a fraction of the Sigma
    execution the plan serves; the interpolatory rule is always ready after
    about a second, so a small budget never refuses.
    """
    value = os.environ.get("LORRAX_UNIFORM_RULE_BUDGET_S", "").strip()
    return float(value) if value else UNIFORM_RULE_TIME_BUDGET_SECONDS


def _uniform_rule_eps():
    """Sup-norm tolerance of the measure-independent box rule, or None.

    Set ``LORRAX_UNIFORM_RULE_EPS`` (relative to the ``1/eta`` kernel peak)
    to route every product window through ``minimax.uniform_rule`` first.
    Unset, the incumbent measure-adapted paths are used unchanged.
    """
    value = os.environ.get("LORRAX_UNIFORM_RULE_EPS", "").strip()
    if not value:
        return None
    eps = float(value)
    if not 0.0 < eps < 1.0:
        raise ValueError("LORRAX_UNIFORM_RULE_EPS must lie in (0, 1)")
    return eps


def _uniform_box_candidate(spec, eta, eps, max_nodes, factor_growth_cap,
                           relative_target):
    """Box rule for one product window: ``(candidate_or_None, metrics)``.

    The rule depends on the window only through the support box of its
    denominators ``d = omega - z`` (fit and validation cells together, padded
    by 2%): ``minimax.uniform_rule`` builds ``1/d ~= sum_k w_k exp(i t_k d)``
    with ``sup |error| <= eps`` on the whole box.  The window's histogram
    (its cell masses) never reaches the builder; it only scores the finished
    rule through the ordinary consumer gates below.

    Why the histogram is kept out.  The measure-adapted fits it replaced
    were accurate where the mass is and nowhere else: a metal's single state
    at E_F at Gamma carries negligible mass and came out 0.95 meV wrong
    (Na, KNOWN_LORRAX_ISSUES 2026-09-01), and their residual gate did not
    predict delivered error.  A sup bound on the support is the same
    statement for every state in the window.

    Tempting, and why not:
    * Weighting the builder's sample cloud by the cell masses "to save
      nodes" re-creates exactly that failure.
    * Using the window's apportioned ``relative_target`` as the builder's
      eps: on windows with a large error envelope it is 50-100x tighter than
      the deck's eps, doubles the node count of the tails and used to force
      a second full reduction (see the gate below).
    * Building only on the fit cells: the campaign's rules were exact on
      the lattice representatives and wrong between them; the validation
      cells (a different lattice) and the raw-corner widening are what make
      the box the support and not a sample of it.
    * Building a second rule for the ``Im d < 0`` windows: ``1/conj(d) =
      conj(1/d)``, so the same rule serves with ``t -> -conj(t)``,
      ``w -> conj(w)``; a second build is the same rule at twice the
      planning wall.

    Cost: the rule is a function of ``(box, eps)`` only, so it is cached by
    the plan cache and would be shareable across decks with the same box.
    """
    from minimax.uniform_rule import build_uniform_rule  # noqa: PLC0415

    d = np.concatenate([
        np.asarray(spec["problem"].denominators, np.complex128).ravel(),
        np.asarray(spec["validation"].denominators, np.complex128).ravel()])
    # The builder assumes ``Im d > 0`` (decay along real time).  Windows on
    # the other omega sign have ``Im d < 0`` everywhere; their rule is the
    # conjugate of the ``Im d > 0`` rule (applied at the end).  A window with
    # both signs cannot be a box rule (it is not one of ours: every product
    # window has a definite pole sign).
    conjugate = bool(np.all(d.imag < 0.0))
    if conjugate:
        d = np.conj(d)
    elif not np.all(d.imag > 0.0):
        return None, None
    re_lo, re_hi = float(np.min(d.real)), float(np.max(d.real))
    support = (re_lo, re_hi)   # the sampled denominators before widening
    # The lattice cells are bin representatives; the box must cover the RAW
    # products state + pole_sign*pole over the window's own frequencies, so
    # widen it to the corner extremes of (raw state energy) x (pole cells).
    # On Na 8x8x8 (25 bins per axis) this never moved an edge
    # (LORRAX_UNIFORM_RULE_TRACE, arm 04); it is kept because it is free and
    # a coarser lattice would need it.  The corners are the extremes of the
    # two factors taken independently, i.e. a superset of the true (state +
    # pole) range; that is deliberate -- a box that is too small silently
    # loses the sup guarantee, one that is too wide only costs nodes.
    try:
        # ``raw_state_energy`` is ``pole_sign * signed`` (see _state_products);
        # the internal sums are ``signed + pole_sign * pole`` =
        # ``pole_sign * (raw + pole)``.  Getting this sign wrong widened the
        # valence boxes across zero (Si arm 15: val:bulk kappa 1.04 -> 66).
        raw = np.asarray(spec["raw_state_energy"], np.float64)
        poles = np.asarray(_selected_pole_values(spec), np.complex128)
        freq = np.asarray(spec["problem"].frequencies, np.float64)
        sign = float(spec["pole_sign"])
        corners = [f - sign * (e + pr)
                   for f in (float(np.min(freq)), float(np.max(freq)))
                   for e in (np.min(raw), np.max(raw))
                   for pr in (np.min(poles.real), np.max(poles.real))]
        # conjugation flips Im d only; the real corners are unchanged
        re_lo, re_hi = min(re_lo, min(corners)), max(re_hi, max(corners))
    except (KeyError, RuntimeError, ValueError, TypeError):
        pass
    # Guard the extremes by 2% of the width, but never let the padding carry
    # a sign-definite edge across zero: that would turn a Laplace box into a
    # crossing box of the full width (measured: val:bulk went from 16 nodes
    # in 3 s to a 414 s fall-through).  The near-zero edge is padded toward
    # zero by at most 30% of its own distance.
    pad = 0.02 * max(re_hi - re_lo, float(eta))
    lo = re_lo - pad if re_lo <= 0.0 else max(re_lo - pad, 0.7 * re_lo)
    hi = re_hi + pad if re_hi >= 0.0 else min(re_hi + pad, 0.7 * re_hi)
    box = (lo, hi, float(np.min(d.imag)), float(np.max(d.imag)))
    box_record = {"support_re": support, "widened_re": (re_lo, re_hi), "box": box}
    if os.environ.get("LORRAX_UNIFORM_RULE_TRACE"):
        print(f"[uniform-box] {spec.get('name', '?')} support_re={support} "
              f"widened_re={(re_lo, re_hi)} box={box} (Ry)")
    rule = build_uniform_rule(
        box, float(eps), time_budget=_uniform_rule_budget())
    times, weights = rule.times, rule.weights
    if conjugate:
        times, weights = -np.conj(times), np.conj(weights)
    # Over the pair ceiling is a refusal of this family, not a reason to
    # tighten anything: the ceiling is a plan-level cap the selector owns.
    if int(times.size) > int(max_nodes):
        return None, None
    evidence = {
        "family": "uniform_box",
        "candidate_tolerance": float(eps),
        "provenance": rule.one_line(),
    }
    try:
        candidate = _rule_candidate(
            spec["problem"], spec["validation"], times, weights, evidence)
    except (RuntimeError, ValueError, FloatingPointError):
        return None, None
    metrics = candidate["metrics"]
    factor = _factor_growth(spec, times, eta)
    # Acceptance.  The box rule's contract is its own sup on the support in
    # the currency that matches Sigma's error: relative error on a
    # sign-definite box, peak-relative on a crossing box (see
    # ``minimax.uniform_rule.build_uniform_rule``; ``rule.sup_error <= eps``
    # is checked by the builder on its finer check cloud and re-checked
    # here).  The incumbent residual clause -- a measure-weighted error
    # relative to the LOCAL 1/|d| -- is recorded but does not gate: with the
    # peak-relative criterion it used to refuse box rules on tails carrying
    # semicore states (rightly: 4 meV at the Na 2s state, arm 09), and with
    # the apportioned allowance as its target it forced a rebuild at an eps
    # 50-100x tighter and a fall-through to the incumbent fit (216 s on the
    # critical rank of a plan whose crossing windows finished in 57 s, Na
    # arm 04).  The runtime-noise clause and the factor-growth cap still
    # gate, the noise clause at the looser of the allowance and eps so a
    # crossing rule with kappa_p99 ~ 50 is not refused at eps 3e-5 where
    # the allowance was 6e-5 (Na arm 07).
    # Tempting, and why not: (1) gating at ``relative_target`` alone "keeps
    # the global contract" -- that contract is the apportionment's; under eps
    # the plan's contract is per-window sup <= eps and the selector budget
    # is lifted accordingly (plan loop).  (2) Dropping the incumbent
    # residual from the record: it is the one measure-weighted number that
    # would show a box that failed to cover its support.
    gate_target = max(float(relative_target), float(eps))
    noise_ok = (metrics[1] * RUNTIME_NOISE_EPSILON
                <= AMPLIFICATION_NOISE_SAFETY * gate_target)
    evidence["incumbent_residual_gate"] = bool(
        _rule_accepted(metrics, gate_target))
    evidence["criterion"] = "relative" if rule.relative else "peak-relative"
    if (rule.sup_error > float(eps) or not noise_ok
            or max(factor) > float(factor_growth_cap)):
        return None, metrics
    required = max(metrics[0], metrics[1] * RUNTIME_NOISE_EPSILON
                   / AMPLIFICATION_NOISE_SAFETY)
    candidate.update(
        required_target=float(required),
        absolute_cost=float(spec["envelope"] * required),
        factor_growth=factor, attempts=[], box_record=box_record)
    return candidate, metrics


def _candidate_rules(spec, eta, max_nodes, factor_growth_cap,
                     relative_target, *, acceptance_rank_margin=None):
    """Return the first on-demand rule passing the gates and rank margin.

    With ``LORRAX_UNIFORM_RULE_EPS`` set the box rule is tried first for
    every window kind and is the only candidate when it passes; the
    incumbent measure-adapted (crossing) and Hackbusch-seeded
    (sign-definite) fits below are reached only on a kappa or
    factor-growth miss.  Tempting, and why not: offering the incumbent
    rules alongside the box rule "so the selector can pick the cheaper
    one" -- the selector minimises nodes under a budget the incumbent
    rules meet on the histogram, not on the support, so it would pick a
    rule with the delivered-error failure the box rule exists to remove
    (Na: 0.95 meV from a 10-node merged roq rule against 0.16 meV).
    """
    eps = _uniform_rule_eps()
    if eps is not None:
        candidate, metrics = _uniform_box_candidate(
            spec, eta, eps, max_nodes, factor_growth_cap, relative_target)
        attempts = []
        if metrics is not None:
            attempts.append({"family": "uniform_box", "candidate_tolerance": float(eps),
                             "refined_residual": metrics[0], "amplification_p99": metrics[1],
                             "amplification_max": metrics[2]})
        # No retry at a tighter eps: the residual clause cannot miss (sup
        # <= eps bounds it) and a kappa or factor-growth miss is a property
        # of the box, which the same box at a tighter eps rebuilds.  A miss
        # falls through to the incumbent paths below with the attempt on
        # record.
        if candidate is not None:
            candidate["attempts"] = attempts
            return [candidate]
    if acceptance_rank_margin is None:
        acceptance_rank_margin = (
            0 if spec["kind"] == "crossing"
            else SIGN_DEFINITE_ACCEPTANCE_RANK_MARGIN)
    acceptance_rank_margin = int(acceptance_rank_margin)
    if acceptance_rank_margin < 0:
        raise ValueError("acceptance_rank_margin must be non-negative")
    if spec["kind"] == "crossing":
        candidate = _measure_adapted_candidate(
            spec, eta, max_nodes, factor_growth_cap, relative_target)
        if candidate is None:
            raise _ProductWindowRefusal(
                f"delivered product window {spec['name']!r} refused: the "
                "measure-adapted fit did not survive the residual, noise, "
                "node, and factor-growth gates")
        return [candidate]

    best_pair = (np.inf, np.inf)
    attempts = []
    first_passing_rank = None
    rule_rows = _sign_definite_candidates(
        spec["problem"], relative_target, max_nodes)
    iterator = iter(rule_rows)
    while True:
        try:
            times, weights, evidence = next(iterator)
        except StopIteration:
            break
        except (FloatingPointError, OverflowError, RuntimeError, ValueError,
                np.linalg.LinAlgError) as exc:
            raise _ProductWindowRefusal(
                f"delivered product window {spec['name']!r} refused: "
                f"{exc}") from exc
        try:
            candidate = _rule_candidate(
                spec["problem"], spec["validation"],
                times, weights, evidence)
            refined = candidate["metrics"]
            factor = _factor_growth(spec, times, eta)
            best_pair = min(best_pair, (refined[0], refined[1]))
            attempts.append({
                "family": evidence["family"],
                "candidate_tolerance": evidence["candidate_tolerance"],
                "node_count": int(times.size),
                "refined_residual": refined[0],
                "amplification_p99": refined[1],
                "amplification_max": refined[2],
                "factor_log_growth_max": max(factor),
            })
            if (not _rule_accepted(refined, relative_target)
                    or max(factor) > float(factor_growth_cap)):
                continue
            if first_passing_rank is None:
                first_passing_rank = int(times.size)
            if int(times.size) < first_passing_rank + acceptance_rank_margin:
                continue
            required_target = max(
                refined[0],
                refined[1] * RUNTIME_NOISE_EPSILON
                / AMPLIFICATION_NOISE_SAFETY)
            candidate.update(
                required_target=float(required_target),
                absolute_cost=float(spec["envelope"] * required_target),
                factor_growth=factor,
                first_passing_rank=int(first_passing_rank),
                acceptance_rank_margin=acceptance_rank_margin,
                attempts=attempts.copy())
            return [candidate]
        except (FloatingPointError, OverflowError, RuntimeError, ValueError,
                np.linalg.LinAlgError) as exc:
            attempts.append({
                "family": evidence.get("family", "unknown"),
                "candidate_tolerance": evidence.get("candidate_tolerance"),
                "refusal": str(exc)})
    fallback = _measure_adapted_candidate(
        spec, eta, max_nodes, factor_growth_cap, relative_target)
    if fallback is not None:
        fallback["attempts"] = attempts
        fallback["evidence"]["provenance"] += (
            "; noncrossing fit missed refined consumer gates")
        return [fallback]
    residual, amplification = best_pair
    raise _ProductWindowRefusal(
        f"delivered product window {spec['name']!r} refused: achieved "
        f"(residual={residual:.6g}, amplification_p99={amplification:.6g}); "
        "the on-demand noncrossing family and its measure-adapted fallback "
        "did not survive the residual, noise, and factor-growth gates",
        float(residual), float(amplification))


def _pointwise_rule_costs(specs, candidates, frequency_count):
    """Return achieved envelope error at every externally served frequency."""
    costs = np.zeros(int(frequency_count), dtype=np.float64)
    for spec, candidate in zip(specs, candidates):
        np.add.at(
            costs, np.asarray(spec["omega_idx"], dtype=np.int64),
            np.asarray(spec["envelope_by_frequency"], dtype=np.float64)
            * float(candidate["required_target"]))
    return costs


def _pointwise_window_allowance(spec, pointwise_budget):
    """Largest relative error this window alone may spend at every omega."""
    positions = np.asarray(spec["omega_idx"], dtype=np.int64)
    envelope = np.asarray(spec["envelope_by_frequency"], dtype=np.float64)
    return min(0.5, float(np.min(
        np.asarray(pointwise_budget, dtype=np.float64)[positions] / envelope)))


def _select_rules(specs, candidates_by_window, total_absolute_budget,
                  pair_ceiling, *, pointwise_budget):
    """Select the minimum-pair plan under every frequency's error budget.

    The exact binary model has one variable per offered rule, so its retained
    state is bounded by the candidate census rather than by a growing Pareto
    frontier.  Each window contributes exactly one rule.
    """
    pointwise_budget = np.asarray(pointwise_budget, dtype=np.float64)
    frequency_count = int(pointwise_budget.size)
    candidate_rows = []
    window_slices = []
    offset = 0
    for spec, candidates in zip(specs, candidates_by_window):
        positions = np.asarray(spec["omega_idx"], dtype=np.int64)
        envelope = np.asarray(
            spec["envelope_by_frequency"], dtype=np.float64)
        start = offset
        for candidate in candidates:
            pointwise_cost = np.zeros(frequency_count, dtype=np.float64)
            np.add.at(
                pointwise_cost, positions,
                envelope * float(candidate["required_target"]))
            candidate_rows.append((
                int(candidate["times"].size), pointwise_cost))
            offset += 1
        window_slices.append(slice(start, offset))

    variable_count = len(candidate_rows)
    matrix = np.zeros(
        (len(window_slices) + 1 + frequency_count, variable_count),
        dtype=np.float64)
    for row, window_slice in enumerate(window_slices):
        matrix[row, window_slice] = 1.0
    nodes_by_candidate = np.asarray(
        [row[0] for row in candidate_rows], dtype=np.float64)
    matrix[len(window_slices)] = nodes_by_candidate
    if frequency_count:
        matrix[len(window_slices) + 1:] = np.asarray(
            [row[1] for row in candidate_rows]).T
    lower = np.concatenate((
        np.ones(len(window_slices)), [-np.inf],
        np.full(frequency_count, -np.inf)))
    upper = np.concatenate((
        np.ones(len(window_slices)), [float(pair_ceiling)],
        pointwise_budget))
    result = milp(
        c=nodes_by_candidate,
        integrality=np.ones(variable_count, dtype=np.int8),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"presolve": True})
    if result.success:
        choices = tuple(
            int(np.argmax(result.x[window_slice]))
            for window_slice in window_slices)
        selected = [candidates[index]
                    for candidates, index in zip(
                        candidates_by_window, choices)]
        nodes = sum(int(candidate["times"].size)
                    for candidate in selected)
        pointwise_cost = _pointwise_rule_costs(
            specs, selected, frequency_count)
        if (nodes > int(pair_ceiling)
                or np.any(pointwise_cost > pointwise_budget)):
            raise AssertionError(
                "integer rule selection violated its pointwise contract")
        feasible = [(nodes, float(np.max(pointwise_cost)), choices,
                     pointwise_cost)]
    else:
        feasible = []

    # Preserve a cheap scalar diagnostic for a budget shortfall.  It does not
    # participate in selection and therefore cannot prune a feasible plan.
    surrogate_frequency = int(np.argmax(pointwise_budget))

    def surrogate_cost(spec, candidate):
        positions = np.asarray(spec["omega_idx"], dtype=np.int64)
        local = np.nonzero(positions == surrogate_frequency)[0]
        if not local.size:
            return 0.0
        envelope = np.asarray(spec["envelope_by_frequency"], dtype=np.float64)
        return float(envelope[local[0]] * candidate["required_target"])

    if not feasible:
        states = {0: (0.0, ())}
        for spec, candidates in zip(specs, candidates_by_window):
            next_states = {}
            for used, (cost, choices) in states.items():
                for index, candidate in enumerate(candidates):
                    nodes = int(candidate["times"].size)
                    new_used = used + nodes
                    if new_used > int(pair_ceiling):
                        continue
                    new_cost = cost + surrogate_cost(spec, candidate)
                    previous = next_states.get(new_used)
                    if previous is None or new_cost < previous[0]:
                        next_states[new_used] = (
                            new_cost, choices + (index,))
            states = next_states
        best = min(states.items(), key=lambda item: item[1][0], default=None)
        blocking = max(
            zip(specs, candidates_by_window),
            key=lambda pair: min(c["absolute_cost"] for c in pair[1]))
        candidate = min(blocking[1], key=lambda row: row["absolute_cost"])
        metrics = candidate["metrics"]
        detail = "no bounded combination"
        best_cost = float("inf")
        blocking_budget = float(total_absolute_budget)
        if best is not None:
            choices = best[1][1]
            selected = [candidates[index] for candidates, index in zip(
                candidates_by_window, choices)]
            costs = _pointwise_rule_costs(
                specs, selected, pointwise_budget.size)
            ratios = np.divide(
                costs, pointwise_budget, out=np.zeros_like(costs),
                where=pointwise_budget > 0.0)
            blocking_frequency = int(np.argmax(ratios))
            best_cost = float(costs[blocking_frequency])
            blocking_budget = float(pointwise_budget[blocking_frequency])
            detail = (f"pointwise cost={best_cost:.6g}, "
                      f"budget={blocking_budget:.6g}, "
                      f"frequency index={blocking_frequency}")
        message = (
            f"delivered product window {blocking[0]['name']!r} refused: "
            f"achieved (residual={metrics[0]:.6g}, "
            f"amplification_p99={metrics[1]:.6g}); {detail}, "
            f"pair ceiling={int(pair_ceiling)}")
        if best is None:
            # Nothing fits under the node ceiling; tightening cannot help.
            raise RuntimeError(message)
        raise _BudgetShortfall(
            message, best_cost, blocking_budget)
    nodes, required_cost, choices, pointwise_cost = min(
        feasible, key=lambda row: (row[0], row[1]))
    selected = [candidates[index]
                for candidates, index in zip(candidates_by_window, choices)]
    selected_envelope = _pointwise_rule_costs(
        specs, [{"required_target": 1.0}] * len(specs),
        pointwise_budget.size)
    spare = np.divide(
        pointwise_budget - pointwise_cost, selected_envelope,
        out=np.full_like(pointwise_budget, np.inf),
        where=selected_envelope > 0.0)
    spare_relative = max(0.0, float(np.min(spare)))
    for spec, candidate in zip(specs, selected):
        candidate["residual_target"] = (
            candidate["required_target"] + spare_relative)
        if not _rule_accepted(candidate["metrics"],
                              candidate["residual_target"]):
            raise AssertionError("selected rule failed its allocated noise budget")
    return selected, int(nodes), float(required_cost)


def _state_products(branch, raw_energy, state_edge, pole_edge):
    crossing = ((branch.space == "cond" and not branch.neg_omega_half)
                or (branch.space == "val" and branch.neg_omega_half))
    if crossing:
        return (
            ("resonant", -np.inf, pole_edge, 0),
            ("state_tail", pole_edge, np.inf, 0),
            ("pole_tail", -np.inf, np.inf, 1),
        )
    return (
        ("bulk", state_edge, np.inf, -1),
        ("resonant", -np.inf, state_edge, 0),
        ("pole_tail", -np.inf, state_edge, 1),
    )


def _pole_bounds(count, lower, upper):
    bounds = np.asarray(
        (lower, upper, -np.inf, -np.inf, np.inf, np.inf),
        dtype=np.float64)
    return np.broadcast_to(bounds, (int(count), 6)).copy()


def build_delivered_sigma_windows(
    Omega_poles_by_branch,
    B_poles_by_branch,
    branches: Sequence[_SigmaBranch],
    omega_grid_ry,
    *,
    regularization_width_ry: float,
    envelope_relative_target: float,
    state_amplitudes_by_branch=None,
    reference_sigma_omega=None,
    max_nodes: int = MAX_WINDOW_TAU_PAIRS,
    lattice_bins: int = DEFAULT_LATTICE_BINS,
    envelope_error_safety: float = ENVELOPE_ERROR_SAFETY,
    factor_growth_cap: float = FACTOR_GROWTH_CAP,
    edge_factor: float = 1.5,
    crossing_eps_q: float = 1.0e-3,
    use_shipped_minimax_tables: bool = True,
    pane_times: tuple = (),
    tau_grid_mode: str = "free",
    measures_by_branch=None,
    mesh_xy=None,
    plan_cache_path=None,
    plan_cache_request_fingerprint=None,
):
    """Build the owner-specified product-window delivered Sigma plan."""
    started = time.perf_counter()
    branch_rows = list(branches)
    omega_grid = np.asarray(omega_grid_ry, dtype=np.float64)
    eta = float(regularization_width_ry)
    target = float(envelope_relative_target)
    safety = float(envelope_error_safety)
    factor_cap = float(factor_growth_cap)
    pair_ceiling = int(max_nodes) if max_nodes is not None else 0
    grid_mode = str(tau_grid_mode).strip().lower()
    del crossing_eps_q, use_shipped_minimax_tables
    if (omega_grid.ndim != 1 or not omega_grid.size
            or not np.all(np.isfinite(omega_grid))):
        raise ValueError("omega_grid_ry must be a nonempty finite vector")
    if not 0.0 < target < 1.0:
        raise ValueError("envelope_relative_target must lie in (0,1)")
    if not 0.0 < safety <= 1.0:
        raise ValueError("envelope_error_safety must lie in (0,1]")
    if pair_ceiling < 1:
        raise ValueError("max_nodes must permit at least one pair")
    if tuple(pane_times):
        raise ValueError("on-demand planning does not accept pane time grids")
    if grid_mode != "free":
        raise ValueError(
            "on-demand planning uses one fitted grid per window; "
            "tau_grid_mode must be 'free'")
    geometry = delivered_product_geometry(
        branch_rows, eta, edge_factor=float(edge_factor))
    split = geometry["pole_edge_ry"]

    census_started = time.perf_counter()
    if measures_by_branch is None:
        omega_rows = _per_branch(
            Omega_poles_by_branch, branch_rows, "Omega_poles_by_branch")
        residue_rows = _per_branch(
            B_poles_by_branch, branch_rows, "B_poles_by_branch")
        amplitude_rows = _optional_per_branch(
            state_amplitudes_by_branch, branch_rows,
            "state_amplitudes_by_branch")
        measure_rows = [
            _pole_measures(
                branch, Omega, B, eta, amplitude, int(lattice_bins), split,
                mesh_xy=mesh_xy)
            for branch, Omega, B, amplitude in zip(
                branch_rows, omega_rows, residue_rows, amplitude_rows)
        ]
    else:
        measure_rows = _per_branch(
            measures_by_branch, branch_rows, "measures_by_branch")
        for branch, measure in zip(branch_rows, measure_rows):
            if not np.isclose(float(measure[7]["pole_split_ry"]), split,
                              rtol=0.0, atol=1.0e-13):
                raise ValueError(
                    f"branch {branch.tag!r} was measured at pole split "
                    f"{measure[7]['pole_split_ry']}, expected {split}")
    census_seconds = time.perf_counter() - census_started

    reference = None
    if reference_sigma_omega is not None:
        reference = np.asarray(reference_sigma_omega, np.complex128)
        if (reference.shape != omega_grid.shape
                or not np.all(np.isfinite(reference))
                or not float(np.max(np.abs(reference))) > 0.0):
            raise ValueError("reference_sigma_omega is invalid")

    window_geometry_started = time.perf_counter()
    specs, branch_reports = [], []
    combined_envelope = np.zeros(omega_grid.size, dtype=np.float64)
    for branch, measure in zip(branch_rows, measure_rows):
        positions = np.asarray(branch.omega_idx, dtype=np.int64)
        frequencies = omega_grid[positions]
        expected = (-np.asarray(branch.omega_abs)
                    if branch.neg_omega_half else np.asarray(branch.omega_abs))
        if not np.allclose(frequencies, expected, rtol=0.0, atol=1.0e-13):
            raise ValueError(
                f"branch {branch.tag!r} frequency indices disagree")
        pole_sign = 1.0 if branch.space == "cond" else -1.0
        raw_energy = pole_sign * np.asarray(measure[0], np.float64)
        report = {
            "tag": branch.tag, "space": branch.space,
            "negative_frequency_half": bool(branch.neg_omega_half),
            "raw_tuple_count": int(measure[6]),
            "live_state_count": int(raw_energy.size),
            "live_pole_count": int(np.asarray(measure[5]).size),
            "window_axis": "state_interval_x_pole_interval",
            "state_support": "plain_interval",
            "plan_start": len(specs), "windows": [], **dict(measure[7]),
        }
        for name, state_lower, state_upper, pole_interval in _state_products(
                branch, raw_energy, geometry["state_edge_ry"], split):
            selected_states = np.nonzero(
                (raw_energy > state_lower) & (raw_energy <= state_upper))[0]
            if not selected_states.size:
                continue
            pole_bounds = (
                (0.0, np.inf) if pole_interval == -1 else
                ((0.0, split) if pole_interval == 0 else
                 (split, np.inf)))
            product = _product_problem(
                selected_states, pole_bounds, measure, frequencies,
                pole_sign, int(lattice_bins))
            if product is None:
                continue
            problem, validation, _pole_indices = product
            routes = _crossing_omega_patches(
                problem, measure, selected_states, pole_bounds, pole_sign,
                geometry["state_edge_ry"], int(lattice_bins))
            patch_count = len(routes)
            for patch_number, (omega_rows, routed_bounds) in enumerate(
                    routes, start=1):
                patch_positions = positions[omega_rows]
                patch_frequencies = frequencies[omega_rows]
                for cell_role, cell_bounds in routed_bounds:
                    cell_product = (_product_problem(
                        selected_states, cell_bounds, measure,
                        patch_frequencies, pole_sign, int(lattice_bins)))
                    if cell_product is None:
                        continue
                    cell_problem, cell_validation, pole_indices = cell_product
                    cell_kind = _window_kind(cell_problem)
                    if patch_count == 1:
                        cell_name = f"{branch.tag}:{name}"
                    else:
                        label = (name if cell_role == "crossing" else
                                 f"{name}:{cell_role}_flank")
                        cell_name = (
                            f"{branch.tag}:{label}"
                            f"[p{patch_number}/{patch_count}]")
                    envelope_by_frequency = (
                        cell_problem.cell_masses[None, :]
                        / np.abs(cell_problem.denominators)
                    ).sum(axis=1)
                    envelope = float(np.max(envelope_by_frequency))
                    selected_raw = raw_energy[selected_states]
                    spec = {
                        "name": cell_name, "branch": branch,
                        "measure": measure, "problem": cell_problem,
                        "validation": cell_validation,
                        "kind": cell_kind, "pole_sign": pole_sign,
                        "pole_interval": pole_interval,
                        "pole_indices": pole_indices,
                        "state_positions": selected_states,
                        "state_indices": np.asarray(measure[2])[selected_states],
                        "raw_state_energy": selected_raw,
                        "state_interval": (float(state_lower), float(state_upper)),
                        "pole_bounds": tuple(map(float, cell_bounds)),
                        "E_ref_A": float(np.min(selected_raw)),
                        "E_ref_B": 0.0,
                        "omega_abs": np.asarray(branch.omega_abs)[omega_rows],
                        "omega_idx": patch_positions,
                        "envelope": envelope,
                        "envelope_by_frequency": envelope_by_frequency,
                        "branch_report": report,
                    }
                    spec["E_ref_A"], spec["E_ref_B"] = _factor_references(spec)
                    specs.append(spec)
                    combined_envelope[patch_positions] += envelope_by_frequency
        report["plan_stop"] = len(specs)
        report["window_count"] = report["plan_stop"] - report["plan_start"]
        branch_reports.append(report)
    window_geometry_seconds = time.perf_counter() - window_geometry_started

    # The ceiling is derived from the measured supports, not taken from the
    # deck: max_nodes is retained only as an optional hard cap for callers who
    # want one.
    derived_ceiling = _derived_pair_ceiling(specs, eta)
    pair_ceiling = (min(derived_ceiling, int(max_nodes))
                    if max_nodes is not None else derived_ceiling)

    combined_scale = float(np.max(combined_envelope))
    pointwise_budget = target * combined_envelope * safety
    total_absolute = float(np.max(pointwise_budget))
    allowances = [_pointwise_window_allowance(spec, pointwise_budget)
                  for spec in specs]
    # Under the box rule the deck's error dial is eps (per-window sup on the
    # support); the apportioned pointwise budget below is the campaign's
    # second dial and is not enforced on selection: with one accepted box
    # rule per window a finite budget can only turn a plan that meets eps
    # everywhere into a "shortfall" and three compounded re-fit stages, each
    # a full reduction cap on the critical rank.  The allowances above still
    # feed the fits (they only loosen the gate, never tighten it past eps).
    select_budget = (np.full_like(pointwise_budget, np.inf)
                     if _uniform_rule_eps() is not None else pointwise_budget)
    cache_fingerprint = _plan_cache_fingerprint(
        specs, eta=eta, target=target, safety=safety,
        factor_cap=factor_cap, pair_ceiling=pair_ceiling,
        grid_mode=grid_mode, lattice_bins=lattice_bins)
    cached = _load_plan_cache(
        plan_cache_path, cache_fingerprint, len(specs))
    cache_status = "disabled" if plan_cache_path is None else "hit"
    window_fit_rows, consolidation_rows = [], []
    window_parallel_seconds = consolidation_seconds = selection_seconds = 0.0
    if cached is not None:
        (fits, free_pairs, required_cost, window_tau_pairs,
         fingerprint_match) = cached
        if not fingerprint_match:
            validated_cost = _validate_cached_fits(
                specs, fits, eta=eta, factor_cap=factor_cap,
                pair_ceiling=pair_ceiling,
                pointwise_budget=select_budget)
            if validated_cost is None:
                cached = None
            else:
                required_cost = validated_cost
                cache_status = "validated_hit"
                _save_plan_cache(
                    plan_cache_path, cache_fingerprint, fits, free_pairs,
                    required_cost, window_tau_pairs)
    if cached is None:
        cache_status = "disabled" if plan_cache_path is None else "miss"
        # Each fit first receives the largest relative allowance it could
        # spend without exceeding the complete plan budget by itself.  The
        # exact selector below then checks the sum of ACHIEVED costs.  This
        # support-derived ceiling avoids a tolerance sweep while preserving
        # the global delivered-error contract.
        # The adapted (rotated) rule per window is fitted ONCE and reused.
        # The exact selector checks their achieved costs.  A shortfall is
        # addressed only by tighter on-demand fits.
        base_specs = specs

        def fit_window(index):
            return _window_candidates_profiled(
                base_specs[index], eta, pair_ceiling, factor_cap,
                allowances[index], adapted_only=True)

        fit_started = time.perf_counter()
        base_candidates, window_fit_rows = _run_parallel_planner_jobs(
            len(base_specs), fit_window, refuse_errors=False)
        window_parallel_seconds += time.perf_counter() - fit_started
        _emit_window_census(
            base_specs, eta, allowances,
            candidates_by_window=base_candidates,
            planner_rows=window_fit_rows, source="live_fit")
        _raise_first_planner_refusal(window_fit_rows)
        fits = None
        consolidation_cache = None
        shortfall = None
        # Four stages, each entered only when the previous cannot close the
        # global budget.  Stage 1 offers the adapted rules alone (this is what
        # every passing deck uses, and it pays for exactly one fit round).
        # Stages 2--4 compound up to three TIGHTENED allowances, which are the
        # only stages that can lower a cost the first two merely re-shuffle.
        tighten_scale = 1.0
        stages = ("adapted",) + ("tightened",) * 3
        for stage_index, stage in enumerate(stages):
            specs = base_specs
            if stage == "adapted":
                candidates_by_window = base_candidates
            elif stage == "tightened":
                # Every window's allowance is the WHOLE plan budget divided by
                # its own envelope, so N windows can each satisfy their own
                # allowance and still sum to N times the budget.  The fits are
                # normally far inside it and the sum fits; when delivered mass
                # moves between windows (an SC map after the spectrum shifts)
                # it does not.  Tighten every allowance by exactly the factor
                # the plan came up short, with a margin, and re-fit: the extra
                # accuracy is bought with nodes instead of a refusal.
                deficit = shortfall.budget / shortfall.best_cost
                step = max(min(deficit * TIGHTEN_MARGIN, 1.0), TIGHTEN_FLOOR)
                # Compounding, because one round only buys back the shortfall
                # it could see: tightening the rules changes which window
                # blocks next (measured on the signed deck: 2.02x over budget
                # blocked by a crossing merge, then 1.34x blocked by a
                # sign-definite tail).
                tighten_scale *= step
                tightened = [min(0.5, allowance * tighten_scale)
                             for allowance in allowances]

                def fit_tight(index):
                    # A window that cannot meet the TIGHTENED allowance is not
                    # a failure: its loose rule is still in the candidate list
                    # below, and the selector may not have needed this window
                    # to give anything up.  Refusing here aborts a plan that
                    # the looser rules can still close (measured: SC map 1
                    # died on 'cond:pole_tail' at residual 1.38e-4 while the
                    # base plan held a rule for it).
                    try:
                        return _window_candidates_profiled(
                            base_specs[index], eta, pair_ceiling, factor_cap,
                            tightened[index], adapted_only=False)
                    except (RuntimeError, ValueError, FloatingPointError,
                            OverflowError, np.linalg.LinAlgError):
                        return [], {"adapted_fit_seconds": 0.0,
                                    "shipped_fallback_seconds": 0.0}

                fit_started = time.perf_counter()
                tight_candidates, tight_rows = _run_parallel_planner_jobs(
                    len(base_specs), fit_tight, refuse_errors=True)
                window_parallel_seconds += time.perf_counter() - fit_started
                window_fit_rows.extend(tight_rows)
                # Keep the looser rules too: they may still be the cheapest
                # choice for a window the shortfall did not come from.
                candidates_by_window = [
                    list(tight) + list(base)
                    for tight, base in zip(tight_candidates, base_candidates)]
            unconsolidated_specs = specs
            unconsolidated_candidates = candidates_by_window
            consolidation_started = time.perf_counter()
            # The tightened stage deliberately does NOT consolidate.  Merging
            # is chosen against the LOOSE candidate set, so re-running it over
            # tightened rules picks a different merge and can price the plan
            # higher than the split one it replaced (measured: 5.65e9 split,
            # 6.58e9 after a tightened re-merge).  Tightening and merging are
            # separate optimisations; letting them interact loses both.
            trial_rows = []
            # Under the box rule there is nothing to merge: the merged trial
            # is fitted on the branch's MEASURE (adapted_only), which is the
            # dependence the box rule removes.  Measured on Na 8x8x8 at
            # P=16 (-5..+5 eV, eps 1e-4): the merged 'val:consolidated' roq
            # rule (10 nodes) replaced three box rules and put the state at
            # E_F at Gamma 0.95 meV off the pane control at omega = 0, while
            # the +-15 eV plan of the same deck, which did not merge, sat at
            # 0.10 meV.
            if stage != "tightened" and _uniform_rule_eps() is None:
                (specs, candidates_by_window, trial_rows,
                 consolidation_cache) = _consolidate_branches(
                    specs, candidates_by_window, eta, pair_ceiling, factor_cap,
                    pointwise_budget, consolidation_cache)
            consolidation_seconds += (
                time.perf_counter() - consolidation_started)
            consolidation_rows.extend(trial_rows)
            selection_started = time.perf_counter()
            try:
                fits, free_pairs, required_cost = _select_rules(
                    specs, candidates_by_window, total_absolute, pair_ceiling,
                    pointwise_budget=select_budget)
                selection_seconds += time.perf_counter() - selection_started
                break
            except RuntimeError as exc:
                selection_seconds += time.perf_counter() - selection_started
                if isinstance(exc, _BudgetShortfall):
                    shortfall = exc
                # Merging leaves the selector ONE rule for that branch, so a
                # merge that is cheaper per node can still make the plan
                # unaffordable with no alternative to fall back on (measured:
                # 'val:consolidated' refused at 3.91e9 against a 2.79e9
                # budget).  Retry un-merged before giving up — the split
                # candidates are already fitted, so this costs only selection.
                if specs is not unconsolidated_specs:
                    selection_started = time.perf_counter()
                    try:
                        fits, free_pairs, required_cost = _select_rules(
                            unconsolidated_specs, unconsolidated_candidates,
                            total_absolute, pair_ceiling,
                            pointwise_budget=select_budget)
                        specs = unconsolidated_specs
                        candidates_by_window = unconsolidated_candidates
                        selection_seconds += (
                            time.perf_counter() - selection_started)
                        break
                    except RuntimeError:
                        selection_seconds += (
                            time.perf_counter() - selection_started)
                if stage == "adapted" and shortfall is None:
                    raise      # ceiling-limited: tightening cannot help
                if stage == "tightened" and stage_index == len(stages) - 1:
                    raise      # all three compounded re-fits were exhausted
        del base_specs

        window_tau_pairs = free_pairs
        _save_plan_cache(
            plan_cache_path, cache_fingerprint, fits, free_pairs,
            required_cost, window_tau_pairs)
    else:
        _emit_window_census(
            specs, eta, allowances,
            candidates_by_window=[[fit] for fit in fits],
            source="fit_cache")

    required_pointwise = _pointwise_rule_costs(
        specs, fits, pointwise_budget.size)
    pointwise_fraction = np.divide(
        required_pointwise, pointwise_budget,
        out=np.zeros_like(required_pointwise), where=pointwise_budget > 0.0)
    output = []
    for spec, fit in zip(specs, fits):
        branch = spec["branch"]
        pole_sign = int(spec["pole_sign"])
        external_sign = -1 if branch.neg_omega_half else 1
        time_exec = pole_sign * np.asarray(fit["times"], np.complex128)
        alpha_exec = (np.asarray(fit["weights"], np.complex128)
                      * np.exp(-eta * time_exec))
        nodes = MinimaxNodes(
            t=jnp.asarray(time_exec, dtype=jnp.complex128),
            alpha=jnp.asarray(alpha_exec, dtype=jnp.complex128))
        state_shape = np.asarray(gather_to_host(branch.E_A)).shape
        mask = np.zeros(int(np.prod(state_shape)), dtype=bool)
        mask[np.asarray(spec["state_indices"], np.int64)] = True
        state_lo, state_hi = spec["state_interval"]
        pole_lo, pole_hi = spec["pole_bounds"]
        metrics = fit["metrics"]
        residual_target = float(fit["residual_target"])
        runtime_noise = metrics[1] * RUNTIME_NOISE_EPSILON
        noise_budget = AMPLIFICATION_NOISE_SAFETY * residual_target
        window = _SigmaWindow(
            name=spec["name"], nodes=nodes,
            mask_A=mask.reshape(state_shape), E_ref_A=spec["E_ref_A"],
            E_ref_B=spec["E_ref_B"],
            omega_sign=pole_sign * external_sign,
            project="full", prefactor=-1.0, max_error=metrics[0],
            provenance=(
                "delivered Cartesian product window; "
                f"residual {metrics[0]:.6g}/{residual_target:.6g}; "
                f"kappa_p99 {metrics[1]:.6g}; runtime noise "
                f"{runtime_noise:.6g}/{noise_budget:.6g}; "
                f"{fit['evidence']['provenance']}"))
        output.append(SharedSigmaWindow(
            window=window, E_A=branch.E_A,
            omega_abs=np.asarray(spec["omega_abs"], np.float64),
            omega_idx=np.asarray(spec["omega_idx"], np.int64),
            pole_indices=spec["pole_indices"],
            bounds=_pole_bounds(len(spec["pole_indices"]), pole_lo, pole_hi),
            phase_real=np.zeros(len(spec["pole_indices"]), dtype=bool),
            band_weight=branch.band_weight))
        spec["branch_report"]["windows"].append({
            "name": spec["name"], "kind": spec["kind"],
            "omega_abs_ry": np.asarray(spec["omega_abs"]).tolist(),
            "omega_indices": np.asarray(spec["omega_idx"]).tolist(),
            "product_state_interval_ry": [state_lo, state_hi],
            "product_pole_interval_ry": [pole_lo, pole_hi],
            "pole_indices": spec["pole_indices"].tolist(),
            "node_count": int(nodes.t.size),
            "relative_residual_target": residual_target,
            "fit_residual": fit["fit_metrics"][0],
            "refined_residual": metrics[0],
            "amplification_p99": metrics[1],
            "amplification_max": metrics[2],
            "runtime_noise_bound": runtime_noise,
            "runtime_noise_budget": noise_budget,
            "noise_budget_met": bool(runtime_noise <= noise_budget),
            "absolute_error_envelope": spec["envelope"],
            "absolute_error_budget": spec["envelope"] * residual_target,
            "green_factor_log_growth_max": fit["factor_growth"][0],
            "screened_factor_log_growth_max": fit["factor_growth"][1],
            "family": fit["evidence"]["family"],
            "candidate_tolerance": fit["evidence"]["candidate_tolerance"],
            "certificate_abs_error_bound": fit["evidence"].get(
                "certificate_abs_error_bound"),
            "catalog_achieved_abs_error": fit["evidence"].get(
                "catalog_achieved_abs_error"),
            "fit_achieved_abs_error": fit["evidence"].get(
                "fit_achieved_abs_error"),
            "requested_range": fit["evidence"].get("requested_range"),
            "fit_range": fit["evidence"].get("fit_range"),
            "fit_provenance": fit["evidence"]["provenance"],
        })

    # Consolidation merges windows AFTER the branch reports recorded their
    # plan slices, so plan_start/plan_stop describe the pre-merge plan and
    # address the wrong rows the moment a merge fires.  Measured: a two-branch
    # plan whose conduction windows merged reported branch 0 as plan[0:2] --
    # picking up the NEXT branch's row, 35 nodes against its own 28 -- and
    # branch 1 as plan[2:3], past the end and empty.  The same slices feed the
    # distinct-tau count below, so that was wrong too whenever a deck merged.
    # Recompute them from the plan that actually shipped.
    report_positions = {}
    for position, spec in enumerate(specs):
        report_positions.setdefault(id(spec["branch_report"]), []).append(
            position)
    for report in branch_reports:
        positions = report_positions.get(id(report), [])
        report["plan_start"] = positions[0] if positions else len(output)
        report["plan_stop"] = (positions[-1] + 1 if positions
                               else report["plan_start"])
        report["window_count"] = len(positions)
        report["node_count"] = sum(
            row["node_count"] for row in report["windows"])
    distinct_tau_count = sum(len({
        (float(value.real), float(value.imag))
        for row in output[report["plan_start"]:report["plan_stop"]]
        for value in np.asarray(row.window.nodes.t)
    }) for report in branch_reports)

    exchange_rate = None
    calibration = "not_calibrated"
    if reference is not None:
        exchange_rate = combined_scale / float(np.max(np.abs(reference)))
        calibration = "calibrated_to_reference_sigma"
    plan_seconds = time.perf_counter() - started
    fit_profile = _planner_rows_profile(window_fit_rows)
    consolidation_profile = _planner_rows_profile(consolidation_rows)
    exchange_seconds = max(
        0.0, window_parallel_seconds
        - fit_profile["critical_rank_fit_seconds"])
    exchange_seconds += max(
        0.0, consolidation_seconds
        - consolidation_profile["critical_rank_fit_seconds"])
    accounted_seconds = (
        census_seconds + window_geometry_seconds
        + fit_profile["critical_rank_adapted_fit_seconds"]
        + fit_profile["critical_rank_shipped_fallback_seconds"]
        + fit_profile["critical_rank_fit_overhead_seconds"]
        + consolidation_profile["critical_rank_fit_seconds"]
        + exchange_seconds + selection_seconds)
    report = {
        "planner": "delivered_product_windows",
        "eta_ry": eta,
        "envelope_relative_target": target,
        "error_currency": "inverse_gap_envelope_relative",
        "physical_relative_sigma_error_claimed": False,
        "envelope_error_safety": safety,
        "planned_absolute_envelope_error_budget": total_absolute,
        "required_absolute_envelope_budget": required_cost,
        "max_pointwise_envelope_budget_fraction": float(
            np.max(pointwise_fraction, initial=0.0)),
        "combined_inverse_gap_envelope": combined_scale,
        "envelope_to_physical_exchange_rate": exchange_rate,
        "exchange_rate_calibration": calibration,
        "lattice_bins_per_axis": int(lattice_bins),
        "amplification_gate": (
            "kappa_p99 * 6.0e-8 <= 0.05 * window_target"),
        "runtime_noise_epsilon": RUNTIME_NOISE_EPSILON,
        "runtime_noise_safety": AMPLIFICATION_NOISE_SAFETY,
        "factor_growth_cap": factor_cap,
        "global_window_tau_pair_ceiling": pair_ceiling,
        "tau_grid_mode": grid_mode,
        "n_windows": len(output),
        "n_tau": window_tau_pairs,
        "window_tau_pairs": window_tau_pairs,
        "distinct_tau_count": distinct_tau_count,
        "plan_seconds": plan_seconds,
        "planning_profile_seconds": {
            "census": census_seconds,
            "window_geometry": window_geometry_seconds,
            "adapted_fits_on_critical_rank": fit_profile[
                "critical_rank_adapted_fit_seconds"],
            "shipped_fallbacks_on_critical_rank": fit_profile[
                "critical_rank_shipped_fallback_seconds"],
            "window_fit_overhead_on_critical_rank": fit_profile[
                "critical_rank_fit_overhead_seconds"],
            "merged_trials_on_critical_rank": consolidation_profile[
                "critical_rank_fit_seconds"],
            "candidate_exchange_and_wait": exchange_seconds,
            "selection": selection_seconds,
            "cache_and_output_assembly": max(
                0.0, plan_seconds - accounted_seconds),
        },
        "planning_parallelism": {
            "process_count": int(process_count()),
            "window_fits": fit_profile,
            "consolidation_trials": consolidation_profile,
            "assignment": "job_index_modulo_process_count",
            "collective": "common.collectives.all_gather_processes",
        },
        "plan_cache_status": cache_status,
        "plan_cache_path": plan_cache_path,
        "plan_cache_fingerprint": cache_fingerprint,
        "branches": branch_reports,
        **geometry,
    }
    _save_complete_delivered_sigma_plan(
        plan_cache_path, plan_cache_request_fingerprint, output, specs,
        report, branch_rows)
    return output, report


__all__ = [
    "AMPLIFICATION_NOISE_SAFETY", "DEFAULT_LATTICE_BINS",
    "ENVELOPE_ERROR_SAFETY", "FACTOR_GROWTH_CAP",
    "MAX_WINDOW_TAU_PAIRS", "RUNTIME_NOISE_EPSILON",
    "build_delivered_sigma_windows",
    "combine_delivered_sigma_pole_measures",
    "delivered_plan_request_fingerprint",
    "delivered_product_geometry",
    "load_complete_delivered_sigma_plan",
    "measure_delivered_sigma_pole_batch",
    "window_candidates",
    "measure_delivered_sigma_pole_fields",
]

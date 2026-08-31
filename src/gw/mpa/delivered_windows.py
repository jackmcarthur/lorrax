"""Plan delivered-error Sigma quadrature on measured product windows.

The planner has two numerical dials: the delivered envelope target and the
retarded broadening ``eta``.  It divides each causal branch into a few
``state interval x pole interval`` windows, measures their weighted reciprocal
problems, and assigns each window part of the global error budget.

Sign-definite windows start from shipped noncrossing minimax tables.  Their
certificate chain is: catalog range and scaled sup-norm bound, physical
rescaling by the smallest gap, achieved error on the fitting lattice, achieved
error on the refined validation lattice, and the runtime-noise gate.  A table
that misses the measured gate is replaced by the next tighter or wider shipped
table.  Crossing windows first try shipped HGL tables and otherwise perform one
deterministic fixed-time weight fit.  The planner never evaluates explicit
state--pole pairs and never emits zero-time or direct terms.

The final acceptance test is

``kappa_p99 * RUNTIME_NOISE_EPSILON <= AMPLIFICATION_NOISE_SAFETY * target``.

The plan also refuses exponent growth above its bounded-factor limit and more
than 200 total ``(window, tau)`` pairs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import os
import pickle
import time
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from common.collectives import (gather_to_host, process_count, process_rank,
                                psum_replicate)
from gw.minimax_screening import MinimaxNodes
from gw.mpa.evaluator import gauss_legendre_interval
from gw.mpa.sigma_windows import SharedSigmaWindow
from gw.ppm_windows import _SigmaBranch, _SigmaWindow
import minimax as _mm
from minimax import (ReciprocalMeasureProblem, solve_fixed_time_weights_fast,
                     tail_refined_lattice_measure)


DEFAULT_LATTICE_BINS = 25
ENVELOPE_ERROR_SAFETY = 0.8
FACTOR_GROWTH_CAP = 30.0
RUNTIME_NOISE_EPSILON = 6.0e-8
AMPLIFICATION_NOISE_SAFETY = 0.05
# Default pair budget for the shipped 0..5 eV demonstration grid; the deck's
# max_nodes OWNS the budget (a 4x-wider omega request physically needs more
# crossing nodes — growth is linear in crossing bandwidth). 200 remains the
# default via the config default; it is a resource certificate, not an
# accuracy dial (dial census 2026-08-31, DERIVE).
MAX_WINDOW_TAU_PAIRS = 200
_PLAN_CACHE_VERSION = 6

# The shipped crossing bundle was generated at eps_q=1e-3.  This value is an
# artifact coordinate, not a planner dial; asking for another value cannot
# select those tables safely.
_CROSSING_EPS_Q = 1.0e-3

# The one crossing fallback uses one deterministic IRLS call.  Twelve steps
# with three-step patience and 20% conditioning slack are the frozen-Na
# settings that keep both crossing windows below their incumbent residuals
# while leaving the complete six-window fitting stage below three seconds.
_CROSSING_FIT_ITERATIONS = 12
_CROSSING_FIT_STALL = 3
_CROSSING_CONDITIONING_SLACK = 0.2

# The causal integral tail is exp(-gamma_min * t_max).  Reserving 90% of the
# requested relative target for that tail is the balanced allocation already
# present in evaluator.damped_rectangle_gauss_rule; it fixes the time interval
# without a search over tail fractions.
_CROSSING_TAIL_FRACTION = 0.9


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
                        int(lattice_bins), _CROSSING_EPS_Q,
                        _CROSSING_FIT_ITERATIONS, _CROSSING_FIT_STALL,
                        _CROSSING_CONDITIONING_SLACK,
                        _CROSSING_TAIL_FRACTION)).encode())
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
                          total_absolute):
    """Re-certify cached nodes and weights on the live measured problems."""
    if len(fits) != len(specs):
        return None
    required_cost = 0.0
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
        required_cost += float(spec["envelope"] * required)
        fit.update(metrics=metrics, factor_growth=factor,
                   required_target=required,
                   absolute_cost=float(spec["envelope"] * required))
    if (sum(int(np.asarray(fit["times"]).size) for fit in fits)
            > int(pair_ceiling)
            or required_cost > float(total_absolute)):
        return None
    return float(required_cost)


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
    """Convert delivered relative error to a uniform absolute kernel bound.

    If ``|Q(d) - 1/d| <= eps_abs`` on every retained cell, the delivered
    numerator at frequency ``i`` is at most ``eps_abs * sum(mass)``.  Thus
    ``relative_target * min(delivered_mass) / sum(mass)`` is a sufficient
    physical absolute target.  It is used only to select a shipped table;
    acceptance is always remeasured in the delivered norm.
    """
    _kept, delivered_mass, _excluded = problem.retained()
    total_mass = float(np.sum(problem.cell_masses))
    target = (float(relative_target) * float(np.min(delivered_mass))
              / total_mass)
    if not np.isfinite(target) or target <= 0.0:
        raise RuntimeError(
            "delivered product window has no positive absolute error target")
    return target


def _catalog_walk(family, range_value, scaled_target, max_nodes, *,
                  target_kind=None, eps_q=None):
    """Return shipped entries in selection order, then tighter/wider order."""
    entries = []
    for entry in _mm.catalog().for_family(family):
        if entry.range_max + 1.0e-12 < float(range_value):
            continue
        if entry.error_bound - 1.0e-18 > float(scaled_target):
            continue
        if entry.node_count > int(max_nodes):
            continue
        if target_kind is not None and entry.target_kind != target_kind:
            continue
        if eps_q is not None and (
                entry.eps_q is None
                or abs(entry.eps_q - float(eps_q)) > 1.0e-12):
            continue
        entries.append(entry)
    return sorted(entries, key=lambda entry: (
        entry.range_max, -entry.error_bound, entry.node_count))


def _load_catalog_entry(entry, *, family, target, eps_q=None):
    """Load one exact catalog entry through the minimax service door."""
    keywords = {} if eps_q is None else {"eps_q": float(eps_q)}
    return _mm.lookup(
        family=family, target=target, range_value=entry.range_max,
        error_bound=entry.error_bound, n_max=entry.node_count, **keywords)


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


def _sign_definite_table_candidates(problem, relative_target, max_nodes):
    """Yield rescaled noncrossing tables in conservative walk order."""
    rotated, transform = _sign_definite_orientation(problem)
    x_min = float(np.min(rotated.real))
    x_max = float(np.max(rotated.real))
    if not 0.0 < x_min <= x_max < np.inf:
        raise RuntimeError(
            f"invalid sign-definite support [{x_min:.6g}, {x_max:.6g}] Ry")
    range_value = x_max / x_min
    absolute_target = _absolute_kernel_target(problem, relative_target)
    scaled_target = absolute_target * x_min
    entries = _catalog_walk(
        "noncrossing", range_value, scaled_target, max_nodes)
    if not entries:
        raise RuntimeError(
            "no shipped noncrossing table covers "
            f"R={range_value:.6g}, scaled target={scaled_target:.6g}, "
            f"and max_nodes={int(max_nodes)}")
    for entry in entries:
        served = _load_catalog_entry(
            entry, family="noncrossing", target="inverse")
        tau = np.asarray(served.nodes, np.float64) / x_min
        alpha = np.asarray(served.weights, np.float64) / x_min
        if transform.startswith("positive"):
            times, weights = 1.0j * tau, alpha
        else:
            times, weights = -1.0j * tau, -alpha
        yield np.asarray(times), np.asarray(weights), {
            "family": "noncrossing",
            "transform": transform,
            "requested_range": range_value,
            "table_range": float(entry.range_max),
            "requested_scaled_error": scaled_target,
            "catalog_error_bound_scaled": float(entry.error_bound),
            "certificate_abs_error_bound": float(entry.error_bound / x_min),
            "catalog_achieved_abs_error": float(served.max_error / x_min),
            "candidate_tolerance": float(entry.error_bound),
            "provenance": served.provenance.one_line(),
        }


def _crossing_geometry(problem, pole_sign):
    oriented = float(pole_sign) * problem.denominators
    gamma = oriented.imag
    if np.any(gamma <= 0.0):
        raise RuntimeError("oriented crossing support is not eta-damped")
    gamma_min = float(np.min(gamma))
    radius = float(np.max(np.abs(oriented.real)))
    return oriented, gamma_min, radius / gamma_min


def _crossing_omega_patches(problem, measure, state_positions, pole_bounds,
                            pole_sign, state_edge, bins):
    """Return the smallest exact omega-patch routing covered by HGL.

    A covered problem returns one identity route.  A wider crossing is split
    into equal contiguous omega patches.  For each patch the original pole
    interval is tiled into a resonant shell and two sign-definite flanks; the
    shell bounds follow directly from that patch's omega rows and the actual
    state extrema.  Problems are rebuilt from the same compact measured cells,
    with no explicit state--pole pairs.  This is also the seam for a future
    user-specified Sigma window with its own omega list and broadening.
    """
    omega_rows = np.arange(problem.frequencies.size, dtype=np.int64)
    entries = [entry for entry in _mm.catalog().for_family("crossing")
               if entry.target_kind == "hgl"
               and entry.eps_q is not None
               and abs(entry.eps_q - _CROSSING_EPS_Q) <= 1.0e-12]
    if not entries:
        raise RuntimeError("the shipped HGL family is empty")
    widest_span = max(float(entry.range_max) for entry in entries)
    if (_window_kind(problem) != "crossing"
            or _crossing_geometry(problem, pole_sign)[2]
            <= widest_span + 1.0e-12):
        return ((omega_rows, (("identity", tuple(map(float, pole_bounds))),)),)

    raw_energy = (float(pole_sign)
                  * np.asarray(measure[0], np.float64))
    selected = raw_energy[np.asarray(state_positions, np.int64)]
    state_min, state_max = float(np.min(selected)), float(np.max(selected))
    original_lo, original_hi = map(float, pole_bounds)
    for patch_count in range(2, omega_rows.size + 1):
        patches = tuple(np.array_split(omega_rows, patch_count))
        planned = []
        for patch in patches:
            oriented = float(pole_sign) * problem.frequencies[patch]
            omega_lo, omega_hi = (float(np.min(oriented)),
                                  float(np.max(oriented)))
            shell_lo = max(
                original_lo, omega_lo - state_max - float(state_edge))
            shell_hi = min(
                original_hi, omega_hi - state_min + float(state_edge))
            shell_bounds = (shell_lo, max(shell_lo, shell_hi))
            shell = _product_problem(
                state_positions, shell_bounds, measure,
                problem.frequencies[patch], pole_sign, int(bins))
            if (shell is not None and _window_kind(shell[0]) == "crossing"
                    and _crossing_geometry(shell[0], pole_sign)[2]
                    > widest_span + 1.0e-12):
                planned = []
                break
            cells = []
            if original_lo < shell_bounds[0]:
                cells.append(("positive", (original_lo, shell_bounds[0])))
            if shell_bounds[0] < shell_bounds[1]:
                cells.append(("crossing", shell_bounds))
            if shell_bounds[1] < original_hi:
                cells.append(("negative", (shell_bounds[1], original_hi)))
            planned.append((patch, tuple(cells)))
        if len(planned) == len(patches):
            return tuple(planned)
    raise RuntimeError(
        "crossing support cannot be served by omega product windows: "
        "even one-row patches exceed the widest shipped HGL span "
        f"A={widest_span:.6g}")


def _crossing_table_candidates(problem, pole_sign, relative_target,
                               max_nodes):
    """Yield HGL table rules whose scaled span and bound cover the support."""
    _oriented, gamma_min, A_dim = _crossing_geometry(problem, pole_sign)
    absolute_target = _absolute_kernel_target(problem, relative_target)
    scaled_target = absolute_target * gamma_min
    entries = _catalog_walk(
        "crossing", A_dim, scaled_target, max_nodes,
        target_kind="hgl", eps_q=_CROSSING_EPS_Q)
    for entry in entries:
        served = _load_catalog_entry(
            entry, family="crossing", target="hgl",
            eps_q=_CROSSING_EPS_Q)
        times = (float(pole_sign) * np.asarray(served.nodes, np.float64)
                 / gamma_min)
        weights = (float(pole_sign) * -1.0j
                   * np.asarray(served.weights, np.float64) / gamma_min)
        yield np.asarray(times), np.asarray(weights), {
            "family": "crossing_hgl",
            "requested_range": A_dim,
            "table_range": float(entry.range_max),
            "requested_scaled_error": scaled_target,
            "catalog_error_bound_scaled": float(entry.error_bound),
            "certificate_abs_error_bound": float(
                entry.error_bound / gamma_min),
            "catalog_achieved_abs_error": float(
                served.max_error / gamma_min),
            "candidate_tolerance": float(entry.error_bound),
            "provenance": served.provenance.one_line(),
        }


def _crossing_fallback_node_count(A_dim, max_nodes):
    """Choose the fallback size from the nearest shipped HGL span."""
    entries = [entry for entry in _mm.catalog().for_family("crossing")
               if entry.target_kind == "hgl"
               and entry.eps_q is not None
               and abs(entry.eps_q - _CROSSING_EPS_Q) <= 1.0e-12]
    if not entries:
        raise RuntimeError("the shipped HGL family is empty")
    lower_ranges = [entry.range_max for entry in entries
                    if entry.range_max <= float(A_dim) + 1.0e-12]
    table_range = (max(lower_ranges) if lower_ranges
                   else min(entry.range_max for entry in entries))
    node_count = max(entry.node_count for entry in entries
                     if entry.range_max == table_range)
    if float(A_dim) > table_range:
        # Crossing node counts grow linearly in bandwidth (measured across
        # the shipped HGL family); a request wider than the widest shipped
        # span extrapolates the density rather than running under-resolved.
        node_count = int(np.ceil(node_count * float(A_dim) / table_range))
    if node_count > int(max_nodes):
        raise RuntimeError(
            f"crossing support A={A_dim:.6g} needs {node_count} fixed "
            f"nodes from the nearest HGL span, max_nodes={int(max_nodes)}")
    return int(node_count), float(table_range)


def _fit_crossing_once(problem, pole_sign, relative_target, max_nodes):
    """Fit one fixed deterministic causal grid; never search node/time pairs."""
    oriented, gamma_min, A_dim = _crossing_geometry(problem, pole_sign)
    # The shipped odd HGL target is certified on ``[-A, A]``, so lookup uses
    # the support radius above.  The unconstrained IRLS fallback keeps its
    # established density against the complete real span; it is not an HGL
    # certificate and reducing that grid changed its conditioning.
    fit_A_dim = float(np.ptp(oriented.real)) / gamma_min
    widest_hgl = max(
        float(entry.range_max)
        for entry in _mm.catalog().for_family("crossing")
        if entry.target_kind == "hgl"
        and entry.eps_q is not None
        and abs(entry.eps_q - _CROSSING_EPS_Q) <= 1.0e-12)
    # A support whose physical HGL radius is covered keeps the established
    # widest-table density even when the conservative end-to-end span is a
    # little larger.  Truly wider supports are patched before this fallback.
    covered_hgl = [
        float(entry.range_max)
        for entry in _mm.catalog().for_family("crossing")
        if entry.target_kind == "hgl"
        and entry.eps_q is not None
        and abs(entry.eps_q - _CROSSING_EPS_Q) <= 1.0e-12
        and entry.range_max <= fit_A_dim + 1.0e-12
    ]
    fallback_A_dim = (
        max(covered_hgl, default=fit_A_dim)
        if A_dim <= widest_hgl + 1.0e-12 else fit_A_dim
    )
    node_count, source_range = _crossing_fallback_node_count(
        fallback_A_dim, max_nodes)
    target = min(float(relative_target), 0.5)
    t_max = (np.log(1.0 / (_CROSSING_TAIL_FRACTION * target))
             / gamma_min)
    tau, positive_weights = gauss_legendre_interval(
        node_count, 0.0, t_max)
    times = float(pole_sign) * np.asarray(tau, np.float64)
    seed = (float(pole_sign) * -1.0j
            * np.asarray(positive_weights, np.float64))
    weights, _objective = solve_fixed_time_weights_fast(
        problem, times,
        iterations=_CROSSING_FIT_ITERATIONS,
        stall_iterations=_CROSSING_FIT_STALL,
        conditioning_slack=_CROSSING_CONDITIONING_SLACK,
        conditioning_pass=True,
        start_weights=seed)
    return np.asarray(times), np.asarray(weights), {
        "family": "crossing_fixed_time_fit",
        "requested_range": A_dim,
        "fit_span": fit_A_dim,
        "node_count_source_range": source_range,
        "candidate_tolerance": float(relative_target),
        "eta_floor_ry": gamma_min,
        "real_span_ry": float(np.ptp(oriented.real)),
        "time_ceiling_ry_inverse": float(t_max),
        "fit_iterations": _CROSSING_FIT_ITERATIONS,
        "fit_stall_iterations": _CROSSING_FIT_STALL,
        "conditioning_slack": _CROSSING_CONDITIONING_SLACK,
        "provenance": "one deterministic fixed-time IRLS fit",
    }


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
                    cells.append(pole_cells[keep] + 1.0j * eta)
    pole_values = np.concatenate(cells)
    screened = float(np.max(np.real(
        -1.0j * pole_values[:, None] * time_exec[None, :])))
    return green, screened


def _candidate_rules(spec, eta, max_nodes, factor_growth_cap,
                     relative_target):
    """Return the first lookup-first rule passing the measured window gates."""
    best_pair = (np.inf, np.inf)
    attempts = []
    if spec["kind"] == "crossing":
        def rules():
            yield from _crossing_table_candidates(
                spec["problem"], spec["pole_sign"], relative_target,
                max_nodes)
            # At most one optimized fit exists: it is made only after every
            # matching shipped HGL table has missed the measured gates.
            yield _fit_crossing_once(
                spec["problem"], spec["pole_sign"], relative_target,
                max_nodes)
        rule_rows = rules()
    else:
        rule_rows = _sign_definite_table_candidates(
            spec["problem"], relative_target, max_nodes)
    iterator = iter(rule_rows)
    while True:
        try:
            times, weights, evidence = next(iterator)
        except StopIteration:
            break
        except (FloatingPointError, OverflowError, RuntimeError, ValueError,
                np.linalg.LinAlgError) as exc:
            raise RuntimeError(
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
            required_target = max(
                refined[0],
                refined[1] * RUNTIME_NOISE_EPSILON
                / AMPLIFICATION_NOISE_SAFETY)
            candidate.update(
                required_target=float(required_target),
                absolute_cost=float(spec["envelope"] * required_target),
                factor_growth=factor,
                attempts=attempts.copy())
            return [candidate]
        except (FloatingPointError, OverflowError, RuntimeError, ValueError,
                np.linalg.LinAlgError) as exc:
            attempts.append({
                "family": evidence.get("family", "unknown"),
                "candidate_tolerance": evidence.get("candidate_tolerance"),
                "refusal": str(exc)})
    residual, amplification = best_pair
    raise RuntimeError(
        f"delivered product window {spec['name']!r} refused: achieved "
        f"(residual={residual:.6g}, amplification_p99={amplification:.6g}); "
        "the shipped product-window family and its one crossing fallback "
        "did not survive the residual, noise, and factor-growth gates")


def _select_rules(specs, candidates_by_window, total_absolute_budget,
                  pair_ceiling):
    """Exact small integer plan: minimum pairs whose budget cost fits."""
    states = {0: (0.0, ())}
    for candidates in candidates_by_window:
        next_states = {}
        for used, (cost, choices) in states.items():
            for index, candidate in enumerate(candidates):
                nodes = int(candidate["times"].size)
                new_used = used + nodes
                if new_used > int(pair_ceiling):
                    continue
                new_cost = cost + candidate["absolute_cost"]
                previous = next_states.get(new_used)
                if previous is None or new_cost < previous[0]:
                    next_states[new_used] = (new_cost, choices + (index,))
        states = next_states
    feasible = [(nodes, cost, choices)
                for nodes, (cost, choices) in states.items()
                if cost <= float(total_absolute_budget)]
    if not feasible:
        best = min(states.items(), key=lambda item: item[1][0], default=None)
        blocking = max(
            zip(specs, candidates_by_window),
            key=lambda pair: min(c["absolute_cost"] for c in pair[1]))
        candidate = min(blocking[1], key=lambda row: row["absolute_cost"])
        metrics = candidate["metrics"]
        detail = "no bounded combination"
        if best is not None:
            detail = (f"best cost={best[1][0]:.6g}, "
                      f"budget={float(total_absolute_budget):.6g}")
        raise RuntimeError(
            f"delivered product window {blocking[0]['name']!r} refused: "
            f"achieved (residual={metrics[0]:.6g}, "
            f"amplification_p99={metrics[1]:.6g}); {detail}, "
            f"pair ceiling={int(pair_ceiling)}")
    nodes, required_cost, choices = min(feasible, key=lambda row: (row[0], row[1]))
    selected = [candidates[index]
                for candidates, index in zip(candidates_by_window, choices)]
    envelope_sum = sum(spec["envelope"] for spec in specs)
    spare_relative = ((float(total_absolute_budget) - required_cost)
                      / envelope_sum)
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
    max_direct_terms: int = 32,
    measures_by_branch=None,
    mesh_xy=None,
    plan_cache_path=None,
    plan_cache_request_fingerprint=None,
):
    """Build the owner-specified product-window delivered Sigma plan."""
    del max_direct_terms
    started = time.perf_counter()
    branch_rows = list(branches)
    omega_grid = np.asarray(omega_grid_ry, dtype=np.float64)
    eta = float(regularization_width_ry)
    target = float(envelope_relative_target)
    safety = float(envelope_error_safety)
    factor_cap = float(factor_growth_cap)
    pair_ceiling = int(max_nodes)
    grid_mode = str(tau_grid_mode).strip().lower()
    if (omega_grid.ndim != 1 or not omega_grid.size
            or not np.all(np.isfinite(omega_grid))):
        raise ValueError("omega_grid_ry must be a nonempty finite vector")
    if not 0.0 < target < 1.0:
        raise ValueError("envelope_relative_target must lie in (0,1)")
    if not 0.0 < safety <= 1.0:
        raise ValueError("envelope_error_safety must lie in (0,1]")
    if pair_ceiling < 1:
        raise ValueError("max_nodes must permit at least one pair")
    if not np.isclose(float(crossing_eps_q), _CROSSING_EPS_Q,
                      rtol=0.0, atol=1.0e-15):
        raise ValueError(
            f"lookup-first crossing tables require eps_q={_CROSSING_EPS_Q:g}")
    if not bool(use_shipped_minimax_tables):
        raise ValueError("lookup-first planning requires shipped minimax tables")
    if tuple(pane_times):
        raise ValueError("lookup-first planning does not accept pane time grids")
    if grid_mode != "free":
        raise ValueError(
            "lookup-first planning uses one served grid per window; "
            "tau_grid_mode must be 'free'")
    geometry = delivered_product_geometry(
        branch_rows, eta, edge_factor=float(edge_factor))
    split = geometry["pole_edge_ry"]

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

    reference = None
    if reference_sigma_omega is not None:
        reference = np.asarray(reference_sigma_omega, np.complex128)
        if (reference.shape != omega_grid.shape
                or not np.all(np.isfinite(reference))
                or not float(np.max(np.abs(reference))) > 0.0):
            raise ValueError("reference_sigma_omega is invalid")

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
                    E_ref = float(np.min(selected_raw))
                    if patch_count > 1 and cell_kind != "crossing":
                        _rotated, transform = _sign_definite_orientation(
                            cell_problem)
                        table_sign = (1.0 if transform.startswith("positive")
                                      else -1.0)
                        if pole_sign * table_sign > 0.0:
                            E_ref = float(np.max(selected_raw))
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
                        "E_ref_A": E_ref,
                        "omega_abs": np.asarray(branch.omega_abs)[omega_rows],
                        "omega_idx": patch_positions,
                        "envelope": envelope, "branch_report": report,
                    }
                    specs.append(spec)
                    combined_envelope[patch_positions] += envelope_by_frequency
        report["plan_stop"] = len(specs)
        report["window_count"] = report["plan_stop"] - report["plan_start"]
        branch_reports.append(report)

    combined_scale = float(np.max(combined_envelope))
    total_absolute = target * combined_scale * safety
    cache_fingerprint = _plan_cache_fingerprint(
        specs, eta=eta, target=target, safety=safety,
        factor_cap=factor_cap, pair_ceiling=pair_ceiling,
        grid_mode=grid_mode, lattice_bins=lattice_bins)
    cached = _load_plan_cache(
        plan_cache_path, cache_fingerprint, len(specs))
    cache_status = "disabled" if plan_cache_path is None else "hit"
    if cached is not None:
        (fits, free_pairs, required_cost, window_tau_pairs,
         fingerprint_match) = cached
        if not fingerprint_match:
            validated_cost = _validate_cached_fits(
                specs, fits, eta=eta, factor_cap=factor_cap,
                pair_ceiling=pair_ceiling, total_absolute=total_absolute)
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
        # Each lookup first receives the largest relative allowance it could
        # spend without exceeding the complete plan budget by itself.  The
        # exact selector below then checks the sum of ACHIEVED costs.  This
        # support-derived ceiling avoids a tolerance sweep while preserving
        # the global delivered-error contract.
        candidates_by_window = [
            _candidate_rules(
                spec, eta, pair_ceiling, factor_cap,
                min(0.5, total_absolute / spec["envelope"]))
            for spec in specs]
        fits, free_pairs, required_cost = _select_rules(
            specs, candidates_by_window, total_absolute, pair_ceiling)

        window_tau_pairs = free_pairs
        _save_plan_cache(
            plan_cache_path, cache_fingerprint, fits, free_pairs,
            required_cost, window_tau_pairs)

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
            E_ref_B=0.0, omega_sign=pole_sign * external_sign,
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
            "fit_provenance": fit["evidence"]["provenance"],
        })

    for report in branch_reports:
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
    report = {
        "planner": "delivered_product_windows",
        "eta_ry": eta,
        "envelope_relative_target": target,
        "error_currency": "inverse_gap_envelope_relative",
        "physical_relative_sigma_error_claimed": False,
        "envelope_error_safety": safety,
        "planned_absolute_envelope_error_budget": total_absolute,
        "required_absolute_envelope_budget": required_cost,
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
        "direct_term_count": 0,
        "plan_seconds": time.perf_counter() - started,
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
    "measure_delivered_sigma_pole_fields",
]

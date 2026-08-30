"""Measure-apportioned hybrid planning for the shared Sigma executor.

For each causal branch the physical denominator is

``d = omega - (E + Omega)`` (conduction) or
``d = omega - (E - Omega)`` (valence).

This module is the production port of DEV-80's ``hybrid_planner.py`` and
``run_study.lattice_measure``. It reduces each addressable pole shard onto
the service's tail-refined cloud-in-cell lattice, partitions explicit
state--leading-pole tuples by delivered-mass quantiles, apportions one
conservative true-error envelope by ``mass * measured inverse-gap
difficulty``, then uses the incumbent damped reciprocal fit for
sign-definite windows and the amplification-capped pane/HGL candidate fit
for crossing windows.

The result is ordinary :class:`gw.mpa.sigma_windows.SharedSigmaWindow` rows.
Their parallel ``state_indices`` / ``pole_indices`` vectors are a disjoint
tuple partition consumed by the established shared Sigma executor.  A small
crossing support that exhausts every bounded quadrature attempt may be marked
``direct``; it is evaluated exactly and is never counted as tau.  The
one-pole GN-PPM reduction is the degenerate state-window case.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import time

import jax
import jax.numpy as jnp
import numpy as np

from common.collectives import (all_gather_processes, gather_to_host,
                                process_rank)
from gw.minimax_screening import MinimaxNodes, solve_phase_minimax_bandwidth
from gw.mpa.evaluator import damped_rectangle_positive_rule
from gw.mpa.sigma_windows import SharedSigmaWindow
from gw.ppm_windows import _SigmaBranch, _SigmaWindow
from minimax import (
    ComplexTimeSearchOptions,
    MeasureWindow,
    ReciprocalMeasureProblem,
    apportion_true_error,
    candidate_time_dictionary,
    evaluate_rule,
    fit_damped_reciprocal,
    fit_phase_bounded_candidates,
    partition_measure_windows,
    rule_amplification,
    solve_fixed_time_weights_fast,
    tail_refined_lattice_measure,
)


DEFAULT_LATTICE_BINS = 25
DEFAULT_AMPLIFICATION_CAP = 10.0
TRUE_ERROR_SAFETY = 0.8
WINDOW_COUNT = {"cond": 4, "val": 3}
CROSSING_CUT_FRACTIONS = (0.45, 0.40, 0.60)


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


def _leading_indices(index, count):
    first = index[0]
    if isinstance(first, slice):
        start, stop, step = first.indices(int(count))
        return np.arange(start, stop, step, dtype=np.int64)
    return np.asarray(first, dtype=np.int64).reshape(-1)


def _local_pole_chunks(Omega, B):
    """Yield ``(global pole indices, Omega, B)`` for local unique shards."""
    if tuple(Omega.shape) != tuple(B.shape) or len(Omega.shape) < 1:
        raise ValueError("per-branch pole and residue arrays must match")
    if isinstance(Omega, jax.Array) != isinstance(B, jax.Array):
        raise ValueError("pole and residue arrays must use the same storage type")
    n_poles = int(Omega.shape[0])
    if not isinstance(Omega, jax.Array):
        if process_rank() == 0:
            yield (np.arange(n_poles, dtype=np.int64),
                   np.asarray(Omega), np.asarray(B))
        return
    if bool(getattr(Omega, "is_fully_replicated", False)):
        if process_rank() == 0:
            yield (np.arange(n_poles, dtype=np.int64),
                   np.asarray(Omega.addressable_data(0)),
                   np.asarray(B.addressable_data(0)))
        return
    shards_O, shards_B = Omega.addressable_shards, B.addressable_shards
    if len(shards_O) != len(shards_B):
        raise ValueError("pole and residue shard counts differ")
    for shard_O, shard_B in zip(shards_O, shards_B):
        if shard_O.index != shard_B.index:
            raise ValueError("pole and residue shard layouts differ")
        yield (_leading_indices(shard_O.index, n_poles),
               np.asarray(shard_O.data), np.asarray(shard_B.data))


def _gather_variable(values, masses):
    """Gather variable-length complex cells from every process."""
    count = np.asarray(all_gather_processes(
        np.asarray([values.size], dtype=np.int64))).reshape(-1)
    width = int(np.max(count, initial=0))
    if width == 0:
        return np.empty(0, np.complex128), np.empty(0, np.float64)
    value_pad = np.zeros(width, dtype=np.complex128)
    mass_pad = np.zeros(width, dtype=np.float64)
    value_pad[:values.size] = values
    mass_pad[:masses.size] = masses
    value_all = np.asarray(all_gather_processes(value_pad))
    mass_all = np.asarray(all_gather_processes(mass_pad))
    return (np.concatenate([value_all[row, :size]
                            for row, size in enumerate(count)]),
            np.concatenate([mass_all[row, :size]
                            for row, size in enumerate(count)]))


def _compress(values, masses, bins):
    if not values.size:
        return (np.empty(0, dtype=np.complex128),
                np.empty(0, dtype=np.float64))
    cells, cell_mass, _refined, _refined_mass = (
        tail_refined_lattice_measure(values, masses, bins_per_axis=int(bins)))
    return cells, cell_mass


def _pole_measures(branch, Omega, B, eta, amplitude, bins, *, pole_offset=0):
    """Build compact pole measures plus executable tuple geometry.

    The spatial pole field is reduced once per leading pole.  Tuple mass and
    representative coordinates are then scalar products of that pole summary
    with each live state.  No per-tuple lattice is retained: on a real 8^3,
    48-band, eight-pole run that would mean roughly 200,000 Python objects and
    hundreds of millions of duplicated lattice cells.  The selected windows
    build their bounded measures once in :func:`_tuple_window_problems`.

    Returned state indices are flat indices into the branch's ``(k, band)``
    carrier; pole indices retain their leading-axis identity in the streamed
    fit store.
    """
    signed_energy, state_mass, state_indices = _branch_states(
        branch, amplitude)
    pole_sign = 1.0 if branch.space == "cond" else -1.0
    n_poles = int(Omega.shape[0])
    pole_offset = int(pole_offset)
    if pole_offset < 0:
        raise ValueError("pole_offset must be nonnegative")
    local_values = [[] for _ in range(n_poles)]
    local_masses = [[] for _ in range(n_poles)]
    bad_B = bad_pole = raw_count = 0

    for pole_indices, Omega_chunk, B_chunk in _local_pole_chunks(Omega, B):
        if int(Omega_chunk.shape[0]) != int(pole_indices.size):
            raise ValueError("pole shard leading extent disagrees with its index")
        for local, pole_index in enumerate(pole_indices):
            omega = np.asarray(Omega_chunk[local], np.complex128).reshape(-1)
            residue = np.asarray(B_chunk[local], np.complex128).reshape(-1)
            finite_B = np.isfinite(residue)
            bad_B += int(np.count_nonzero(~finite_B))
            live = finite_B & (np.abs(residue) > 0.0)
            finite_O = np.isfinite(omega)
            gamma = -omega.imag
            bad_pole += int(np.count_nonzero(
                live & (~finite_O | (omega.real <= 0.0) | (gamma < 0.0))))
            live &= finite_O & (omega.real > 0.0) & (gamma >= 0.0)
            if not np.any(live):
                continue
            broadened = omega[live].real - 1.0j * (gamma[live] + eta)
            cells, masses = _compress(
                broadened, np.abs(residue[live]), bins)
            local_values[int(pole_index)].append(cells)
            local_masses[int(pole_index)].append(masses)
            raw_count += int(np.count_nonzero(live)) * int(signed_energy.size)

    bad = np.asarray(all_gather_processes(
        np.asarray([bad_B, bad_pole], dtype=np.int64))).reshape(-1, 2).sum(axis=0)
    if int(bad[0]):
        raise ValueError(
            f"delivered Sigma poles contain {int(bad[0])} nonfinite residues")
    if int(bad[1]):
        raise ValueError(
            f"delivered Sigma poles contain {int(bad[1])} unsupported live poles")

    pole_cells, pole_weights = [], []
    for pole_index in range(n_poles):
        values = (np.concatenate(local_values[pole_index])
                  if local_values[pole_index] else np.empty(0, np.complex128))
        masses = (np.concatenate(local_masses[pole_index])
                  if local_masses[pole_index] else np.empty(0, np.float64))
        values, masses = _gather_variable(values, masses)
        if not values.size:
            pole_cells.append(None)
            pole_weights.append(None)
            continue
        cells, weights = _compress(values, masses, bins)
        pole_cells.append(cells)
        pole_weights.append(weights)

    tuple_mass, representative = [], []
    tuple_state_indices, tuple_pole_indices = [], []
    # State-major, pole-minor is the DEV-80 tuple convention.  Keeping this
    # order makes the planner's membership vectors directly comparable with
    # the hardened toy receipts and deterministic across storage shardings.
    for energy, mass, state_index in zip(
            signed_energy, state_mass, state_indices):
        for pole_index, (cells, weights) in enumerate(zip(
                pole_cells, pole_weights)):
            if cells is None:
                continue
            total_pole_mass = float(np.sum(weights))
            pole_mean = complex(np.sum(weights * cells) / total_pole_mass)
            total = float(mass * total_pole_mass)
            tuple_mass.append(total)
            representative.append(complex(
                energy + pole_sign * pole_mean))
            tuple_state_indices.append(int(state_index))
            tuple_pole_indices.append(int(pole_offset + pole_index))

    raw_global = int(np.asarray(all_gather_processes(
        np.asarray([raw_count], dtype=np.int64))).sum())
    return (
        signed_energy, state_mass, state_indices,
        tuple(pole_cells), tuple(pole_weights),
        np.asarray(tuple_mass, np.float64),
        np.asarray(representative, np.complex128),
        np.asarray(tuple_state_indices, np.int32),
        np.asarray(tuple_pole_indices, np.int32),
        raw_global,
    )


def measure_delivered_sigma_pole_batch(
    branch, Omega, B, *, regularization_width_ry, state_amplitude=None,
    lattice_bins=DEFAULT_LATTICE_BINS, pole_offset=0,
):
    """Reduce one resident leading-pole batch to bounded host measures.

    The returned tuple retains global leading-pole indices through
    ``pole_offset`` while the bounded pole-cell tables remain local to this
    batch. Combine consecutive results with
    :func:`combine_delivered_sigma_pole_measures` before planning.
    """
    return _pole_measures(
        branch, Omega, B, float(regularization_width_ry), state_amplitude,
        int(lattice_bins), pole_offset=int(pole_offset))


def combine_delivered_sigma_pole_measures(batch_measures):
    """Combine consecutive measured pole batches in deterministic order."""
    rows = list(batch_measures)
    if not rows:
        raise ValueError("delivered Sigma needs at least one measured pole batch")
    first = rows[0]
    signed_energy = np.asarray(first[0], np.float64)
    state_mass = np.asarray(first[1], np.float64)
    state_indices = np.asarray(first[2], np.int32)
    pole_cells, pole_weights = [], []
    masses, representative = [], []
    tuple_state_indices, tuple_pole_indices = [], []
    raw_count = 0
    for row in rows:
        if (not np.array_equal(np.asarray(row[0]), signed_energy)
                or not np.array_equal(np.asarray(row[1]), state_mass)
                or not np.array_equal(np.asarray(row[2]), state_indices)):
            raise ValueError(
                "delivered pole batches disagree about their branch states")
        pole_cells.extend(row[3])
        pole_weights.extend(row[4])
        masses.append(np.asarray(row[5], np.float64))
        representative.append(np.asarray(row[6], np.complex128))
        tuple_state_indices.append(np.asarray(row[7], np.int32))
        tuple_pole_indices.append(np.asarray(row[8], np.int32))
        raw_count += int(row[9])

    masses = np.concatenate(masses)
    representative = np.concatenate(representative)
    tuple_state_indices = np.concatenate(tuple_state_indices)
    tuple_pole_indices = np.concatenate(tuple_pole_indices)
    order = np.lexsort((tuple_pole_indices, tuple_state_indices))
    return (
        signed_energy, state_mass, state_indices,
        tuple(pole_cells), tuple(pole_weights),
        masses[order], representative[order],
        tuple_state_indices[order], tuple_pole_indices[order], raw_count,
    )


def _single_measure_window(index, value, mass, frequencies):
    omega_lo, omega_hi = float(np.min(frequencies)), float(np.max(frequencies))
    kind = ("below" if value.real < omega_lo else
            "above" if value.real > omega_hi else "crossing")
    distance = (omega_lo - value.real if kind == "below" else
                value.real - omega_hi if kind == "above" else 0.0)
    scale = float(np.hypot(distance, abs(value.imag)))
    return MeasureWindow(
        name=kind, kind=kind,
        member_indices=np.asarray([index], dtype=np.int64),
        delivered_mass=float(mass), mass_fraction=1.0,
        scale_min=scale, scale_max=scale, scale_span=1.0)


def _partition_tuples(representative, masses, frequencies, space, *,
                      force_single=False):
    live = np.nonzero(masses > 0.0)[0]
    if live.size == 0:
        raise ValueError("delivered Sigma branch has no live tuples")
    if live.size == 1 or bool(force_single):
        if live.size > 1:
            total = float(np.sum(masses[live]))
            value = complex(
                np.sum(masses[live] * representative[live]) / total)
            only = _single_measure_window(0, value, total, frequencies)
            return (replace(
                only, name=f"{only.name}_one_pole",
                member_indices=live),)
        return (_single_measure_window(
            int(live[0]), representative[live[0]], masses[live[0]],
            frequencies),)
    requested = min(int(WINDOW_COUNT[space]), int(live.size))
    try:
        local = partition_measure_windows(
            representative[live], masses[live], frequencies,
            window_count=requested)
    except RuntimeError:
        # A single atom can carry more than one requested quantile's mass,
        # leaving every other label empty. Dropping empty bins is required;
        # the executable and conservative fallback is one window containing
        # every live tuple, not an invented split of that atom.
        total = float(np.sum(masses[live]))
        value = complex(np.sum(masses[live] * representative[live]) / total)
        only = _single_measure_window(0, value, total, frequencies)
        return (replace(only, name=f"{only.name}_tuples",
                        member_indices=live),)
    return tuple(replace(
        window, member_indices=live[window.member_indices],
        name=f"{window.name}_tuples") for window in local)


def _tuple_window_problems(
    member_indices, tuple_state_indices, tuple_pole_indices,
    state_indices, signed_energy, state_mass, pole_cells, pole_weights,
    frequencies, pole_sign, bins,
):
    """Build base/refined lattices once for one executable tuple window.

    For each pole, selected state energies are first compressed to the same
    bounded lattice vocabulary as the spatial residue field.  Their outer sum
    has at most ``bins * bins^2`` cells per pole instead of one copied pole
    lattice per actual ``(k, band, pole)`` tuple.  Tuple membership itself is
    unchanged and remains exact in the executor.
    """
    members = np.asarray(member_indices, np.int64)
    states = np.asarray(tuple_state_indices[members], np.int32)
    poles = np.asarray(tuple_pole_indices[members], np.int32)
    if not members.size:
        raise ValueError("delivered Sigma window has no measured tuple support")

    internal_rows, delivered_rows = [], []
    for pole in np.unique(poles):
        selected_states = states[poles == pole]
        positions = np.searchsorted(state_indices, selected_states)
        if (np.any(positions >= state_indices.size)
                or not np.array_equal(state_indices[positions], selected_states)):
            raise AssertionError(
                "delivered tuple state index is outside its live-state table")
        state_cells, state_weights = _compress(
            np.asarray(signed_energy[positions], np.complex128),
            np.asarray(state_mass[positions], np.float64), bins)
        cells = pole_cells[int(pole)]
        weights = pole_weights[int(pole)]
        if cells is None or weights is None:
            raise AssertionError(
                "delivered tuple selected a pole without measured support")
        internal_rows.append((
            state_cells[:, None] + pole_sign * cells[None, :]).reshape(-1))
        delivered_rows.append((
            state_weights[:, None] * weights[None, :]).reshape(-1))

    internal = np.concatenate(internal_rows)
    delivered = np.concatenate(delivered_rows)
    cells_b, mass_b, cells_r, mass_r = tail_refined_lattice_measure(
        internal, delivered, bins_per_axis=int(bins))
    return (
        ReciprocalMeasureProblem(
            frequencies=frequencies, internal_sums=cells_b,
            cell_masses=mass_b),
        ReciprocalMeasureProblem(
            frequencies=frequencies, internal_sums=cells_r,
            cell_masses=mass_r),
    )


def _window_kind(problem):
    real = problem.denominators.real
    if np.all(real > 0.0):
        return "sign_definite_positive"
    if np.all(real < 0.0):
        return "sign_definite_negative"
    return "crossing"


def _rule_metrics(problem, times, weights):
    denominator = problem.denominators
    relative = np.abs(
        1.0 - denominator * evaluate_rule(times, weights, denominator))
    p99, peak = rule_amplification(times, weights, problem)
    return float(np.max(relative)), float(p99), float(peak)


def _fit_sign_definite(problem, validation, target, max_nodes, amp_cap):
    denominator = problem.denominators
    if np.all(denominator.real > 0.0):
        if np.all(denominator.imag <= 0.0):
            rotated, transform = denominator, "positive_lower"
        elif np.all(denominator.imag >= 0.0):
            rotated, transform = np.conj(denominator), "positive_upper"
        else:
            raise RuntimeError("sign-definite support crosses the real axis")
    elif np.all(denominator.real < 0.0):
        if np.all(denominator.imag >= 0.0):
            rotated, transform = -denominator, "negative_upper"
        elif np.all(denominator.imag <= 0.0):
            rotated, transform = -np.conj(denominator), "negative_lower"
        else:
            raise RuntimeError("sign-definite support crosses the real axis")
    else:
        raise RuntimeError("sign-definite fitter received a crossing window")
    gamma = -rotated.imag
    rectangle = np.asarray([[
        float(np.min(rotated.real)), float(np.max(rotated.real)),
        float(np.min(gamma)), float(np.max(gamma)),
    ]])
    attempts, accepted, best = [], [], None
    for tightening in (1.0, 0.3, 0.1):
        fit = fit_damped_reciprocal(
            rectangle, target_error=max(float(target) * tightening, 1.0e-12),
            max_rank=min(128, int(max_nodes)), training_points=16,
            validation_points=80, contour_count=7, lawson_iterations=6)
        if transform == "positive_lower":
            times, weights = 1.0j * fit.nodes, fit.weights
        elif transform == "positive_upper":
            times, weights = 1.0j * np.conj(fit.nodes), np.conj(fit.weights)
        elif transform == "negative_upper":
            times, weights = -1.0j * fit.nodes, -fit.weights
        else:
            times, weights = -1.0j * np.conj(fit.nodes), -np.conj(fit.weights)
        base = _rule_metrics(problem, times, weights)
        check = _rule_metrics(validation, times, weights)
        row = (check[0], check[1], int(times.size), times, weights, base, check)
        attempts.append({
            "tightening": tightening, "node_count": int(times.size),
            "fit_residual": base[0], "refined_residual": check[0],
            "amplification_p99": check[1], "amplification_max": check[2],
        })
        if best is None or row[:3] < best[:3]:
            best = row
        if check[0] <= target and check[1] <= amp_cap:
            accepted.append(row)
            break
    chosen = min(accepted, key=lambda row: row[:3]) if accepted else best
    if chosen[0] > target or chosen[1] > amp_cap:
        raise RuntimeError(
            "hybrid sign-definite fit missed its apportioned target or "
            f"amplification cap: residual={chosen[0]:.3e}, "
            f"p99={chosen[1]:.3g}, target={target:.3e}, cap={amp_cap:g}")
    return chosen[3], chosen[4], {
        "family": "damped_reciprocal", "transform": transform,
        "fit_residual": chosen[5][0], "refined_residual": chosen[6][0],
        "amplification_p99": chosen[6][1],
        "amplification_max": chosen[6][2], "attempts": attempts,
    }


def _crossing_candidates(problem, target, eta, max_nodes, eps_q,
                         use_shipped, pane_times):
    options = ComplexTimeSearchOptions(
        target_error=max(float(target), 1.0e-12),
        max_nodes=max(8, int(max_nodes)), sector_shortcut=False,
        return_best_on_miss=True)
    dictionary, families = candidate_time_dictionary(problem, options)
    families = np.asarray(families)
    service_mask = ((np.abs(dictionary.imag) <= 1.0e-12)
                    & (dictionary.real > 0.0))
    service = dictionary[service_mask]
    receipt = {
        "service_positive_real_count": int(service.size),
        "service_positive_real_families": sorted(set(
            families[service_mask].tolist())),
    }
    if pane_times:
        pane = np.asarray(pane_times, dtype=np.complex128)
        receipt["pane_provenance"] = "caller-supplied incumbent pane family"
        incumbent = None
    else:
        denominator = problem.denominators
        gamma = denominator.imag
        if np.any(gamma <= 0.0):
            raise RuntimeError(
                "oriented crossing support is not in the upper half-plane")
        mpa_rule = damped_rectangle_positive_rule(
            float(np.min(gamma)), float(np.max(gamma)),
            float(np.max(np.abs(denominator.real))),
            rel_tol=max(float(target), 1.0e-12), max_nodes=int(max_nodes))
        mpa_times = np.asarray(mpa_rule["t"], dtype=np.float64)
        incumbent = (
            mpa_times.astype(np.complex128),
            -1.0j * np.asarray(mpa_rule["h"], dtype=np.float64),
        )
        bandwidth = min(24.0, max(
            1.0e-12, 2.0 * float(np.max(np.abs(denominator.real))) / eta))
        pane_rule = solve_phase_minimax_bandwidth(
            bandwidth, target_error=max(float(target), 1.0e-12),
            max_nodes=max(8, min(500, int(max_nodes))), eps_q=float(eps_q),
            target_kind="hgl", use_shipped_tables=bool(use_shipped))
        hgl_times = np.asarray(pane_rule.tau, np.float64) / eta
        pane = np.concatenate((mpa_times, hgl_times))
        receipt["pane_provenance"] = {
            "mpa": str(mpa_rule["rule_type"]),
            "hgl": pane_rule.provenance,
        }
        receipt["mpa_incumbent_node_count"] = int(mpa_times.size)
        receipt["pane_bandwidth"] = bandwidth
    pane = np.asarray(pane, dtype=np.complex128).reshape(-1)
    pane = pane[(np.abs(pane.imag) <= 1.0e-12) & (pane.real > 0.0)]
    receipt["pane_candidate_count"] = int(pane.size)
    return np.concatenate((pane, service)), pane, incumbent, receipt


def _fit_crossing(problem, validation, target, pole_sign, eta, max_nodes,
                  amp_cap, eps_q, use_shipped, pane_times):
    oriented = ReciprocalMeasureProblem(
        frequencies=pole_sign * problem.frequencies,
        internal_sums=pole_sign * problem.internal_sums,
        cell_masses=problem.cell_masses)
    candidates, pane, incumbent, receipt = _crossing_candidates(
        oriented, target, eta, max_nodes, eps_q, use_shipped, pane_times)
    attempts, accepted, best = [], [], None
    if incumbent is not None:
        times = pole_sign * incumbent[0]
        weights = pole_sign * incumbent[1]
        base = _rule_metrics(problem, times, weights)
        check = _rule_metrics(validation, times, weights)
        row = (check[0], check[1], int(times.size), times, weights,
               base, check, "mpa_positive_incumbent")
        attempts.append({
            "candidate_family": "mpa_positive_incumbent",
            "tightening": 1.0, "node_count": int(times.size),
            "fit_residual": base[0], "refined_residual": check[0],
            "amplification_p99": check[1],
            "amplification_max": check[2],
            "fit_target_met": bool(check[0] <= target),
        })
        best = row
        if check[0] <= target and check[1] <= amp_cap:
            accepted.append(row)
    for family, values in (("pane_plus_service", candidates),
                           ("pane_only_refit", pane)):
        if values.size < 2:
            continue
        for tightening in (1.0, 0.3, 0.1):
            fit = fit_phase_bounded_candidates(
                oriented, values,
                target_error=max(float(target) * tightening, 1.0e-12),
                phase=-1.0j, max_rank=min(200, int(max_nodes)),
                training_frequency_count=min(24, oriented.frequencies.size),
                training_cell_count=420, lawson_iterations=12)
            times = pole_sign * fit.time_nodes
            weights = pole_sign * fit.weights
            base = _rule_metrics(problem, times, weights)
            check = _rule_metrics(validation, times, weights)
            row = (check[0], check[1], int(times.size), times, weights,
                   base, check, fit)
            attempts.append({
                "candidate_family": family, "tightening": tightening,
                "node_count": int(times.size), "fit_residual": base[0],
                "refined_residual": check[0],
                "amplification_p99": check[1],
                "amplification_max": check[2],
                "fit_target_met": bool(fit.target_met),
            })
            if best is None or row[:3] < best[:3]:
                best = row
            if check[0] <= target and check[1] <= amp_cap:
                accepted.append(row)
                break
        if any(row[7] != "mpa_positive_incumbent" for row in accepted):
            break
    if best is None:
        raise RuntimeError("hybrid crossing fit had fewer than two candidates")
    chosen = min(accepted, key=lambda row: row[:3]) if accepted else best
    if chosen[0] > target or chosen[1] > amp_cap:
        raise RuntimeError(
            "hybrid crossing fit missed its apportioned target or "
            f"amplification cap: residual={chosen[0]:.3e}, "
            f"p99={chosen[1]:.3g}, target={target:.3e}, cap={amp_cap:g}")
    return chosen[3], chosen[4], {
        "family": (chosen[7] if isinstance(chosen[7], str)
                   else chosen[7].__class__.__name__),
        "fit_residual": chosen[5][0], "refined_residual": chosen[6][0],
        "amplification_p99": chosen[6][1],
        "amplification_max": chosen[6][2], "attempts": attempts,
        **receipt,
    }


def _crossing_energy_parts(spec, lower_mass_fraction):
    """Split one crossing tuple window at a stable delivered-mass quantile."""
    members = np.asarray(spec["member_indices"], dtype=np.int64)
    masses = np.asarray(spec["tuple_masses"], dtype=np.float64)[members]
    coordinate = np.asarray(
        spec["tuple_representative"], dtype=np.complex128)[members].real
    order = np.argsort(coordinate, kind="stable")
    ordered_members = members[order]
    ordered_mass = masses[order]
    cumulative = np.cumsum(ordered_mass)
    midpoint = (cumulative - 0.5 * ordered_mass) / float(cumulative[-1])
    lower = ordered_members[midpoint < float(lower_mass_fraction)]
    upper = ordered_members[midpoint >= float(lower_mass_fraction)]
    if lower.size == 0 or upper.size == 0:
        raise RuntimeError("crossing energy quantile produced an empty part")
    if not np.array_equal(
            np.sort(np.concatenate((lower, upper))), np.sort(members)):
        raise RuntimeError("crossing energy split lost or duplicated membership")
    return lower, upper


def _aligned_frequency_parts(spec, tuple_parts):
    """Split the owning signed frequency block between ordered tuple parts."""
    representative = np.asarray(
        spec["tuple_representative"], dtype=np.complex128)
    boundary = 0.5 * (
        float(np.max(representative[tuple_parts[0]].real))
        + float(np.min(representative[tuple_parts[1]].real)))
    frequencies = np.asarray(spec["problem"].frequencies, dtype=np.float64)
    positions = np.arange(frequencies.size, dtype=np.int64)
    panes = (positions[frequencies < boundary],
             positions[frequencies >= boundary])
    if any(pane.size == 0 for pane in panes):
        raise RuntimeError("aligned crossing split produced an empty frequency pane")
    if not np.array_equal(
            np.sort(np.concatenate(panes)), np.arange(frequencies.size)):
        raise RuntimeError("aligned frequency panes do not partition omega")
    return panes, boundary


def _crossing_block_spec(parent, members, frequency_positions, name):
    """Build one executable tuple/frequency block from its parent measure."""
    context = parent["measure_context"]
    positions = np.asarray(frequency_positions, dtype=np.int64)
    frequencies = np.asarray(parent["problem"].frequencies)[positions]
    problem, validation = _tuple_window_problems(
        members, *context, frequencies,
        1.0 if parent["branch"].space == "cond" else -1.0,
        int(parent["lattice_bins"]))
    envelope_by_frequency = (
        problem.cell_masses[None, :] / np.abs(problem.denominators)
    ).sum(axis=1)
    envelope = float(np.max(envelope_by_frequency))
    mass = float(np.sum(problem.cell_masses))
    window = replace(
        parent["window"], name=name, kind=_window_kind(problem),
        member_indices=np.asarray(members, dtype=np.int64),
        delivered_mass=mass,
        mass_fraction=(mass / parent["window"].delivered_mass))
    tuple_states = np.asarray(parent["tuple_state_indices"], dtype=np.int32)
    tuple_poles = np.asarray(parent["tuple_pole_indices"], dtype=np.int32)
    return {
        **parent,
        "problem": problem,
        "validation": validation,
        "window": window,
        "member_indices": np.asarray(members, dtype=np.int64),
        "state_indices": tuple_states[members],
        "pole_indices": tuple_poles[members],
        "omega_positions": np.asarray(
            parent["omega_positions"], dtype=np.int64)[positions],
        "envelope": envelope,
        "difficulty": envelope / mass,
        "hardening_parent": parent["window"].name,
    }


def _fit_crossing_blocks(
    spec, budget, *, pole_sign, eta, max_nodes, amp_cap, crossing_eps_q,
    use_shipped_minimax_tables, pane_times,
):
    """Harden a refused crossing window with aligned 2x2 execution blocks.

    Frequency panes are disjoint.  Within each pane the parent absolute
    allowance is split between the two tuple parts in proportion to their
    measured inverse-gap envelopes, so summing the two certified block errors
    cannot exceed the parent allowance.  A refused crossing child may become
    exact direct work only when that child's explicit tuple support fits the
    declared node-equivalent ceiling.
    """
    trials = []
    for fraction in CROSSING_CUT_FRACTIONS:
        try:
            tuple_parts = _crossing_energy_parts(spec, fraction)
            frequency_parts, boundary = _aligned_frequency_parts(
                spec, tuple_parts)
        except RuntimeError as exc:
            trials.append({
                "lower_mass_fraction": fraction, "accepted": False,
                "refusal": str(exc),
            })
            continue
        child_specs, child_fits = [], []
        accepted = True
        refusal = None
        pane_receipts = []
        for pane_index, frequency_positions in enumerate(frequency_parts):
            pane_specs = [
                _crossing_block_spec(
                    spec, members, frequency_positions,
                    (f"{spec['window'].name}.m{fraction:.2f}."
                     f"o{pane_index}.t{tuple_index}"))
                for tuple_index, members in enumerate(tuple_parts)
            ]
            scores = np.asarray(
                [child["envelope"] for child in pane_specs], np.float64)
            shares = scores / float(np.sum(scores))
            pane_rows = []
            for tuple_index, (child, share) in enumerate(
                    zip(pane_specs, shares)):
                child_budget = replace(
                    budget, name=child["window"].name,
                    delivered_mass=child["window"].delivered_mass,
                    measured_difficulty=child["difficulty"],
                    apportionment_weight=(
                        budget.apportionment_weight * float(share)),
                    absolute_error_budget=(
                        budget.absolute_error_budget * float(share)))
                residual_target = (
                    child_budget.absolute_error_budget / child["envelope"])
                direct = False
                try:
                    if child["window"].kind == "crossing":
                        times, weights, evidence = _fit_crossing(
                            child["problem"], child["validation"],
                            residual_target, pole_sign, eta, int(max_nodes),
                            amp_cap, float(crossing_eps_q),
                            bool(use_shipped_minimax_tables), tuple(pane_times))
                    else:
                        times, weights, evidence = _fit_sign_definite(
                            child["problem"], child["validation"],
                            residual_target, int(max_nodes), amp_cap)
                except RuntimeError as exc:
                    n_direct = int(child["state_indices"].size)
                    if (child["window"].kind != "crossing"
                            or n_direct > int(max_nodes)
                            or not str(exc).startswith(
                                "hybrid crossing fit missed")):
                        accepted = False
                        refusal = (
                            f"{child['window'].name} ({n_direct} tuples): {exc}")
                        break
                    direct = True
                    times = np.empty(0, dtype=np.complex128)
                    weights = np.empty(0, dtype=np.complex128)
                    evidence = {
                        "family": "exact_direct_reciprocal_fallback",
                        "fit_residual": 0.0, "refined_residual": 0.0,
                        "amplification_p99": 1.0,
                        "amplification_max": 1.0,
                        "quadrature_refusal": str(exc),
                        "direct_term_count": n_direct,
                    }
                evidence = {
                    **evidence,
                    "crossing_hardening": "aligned_mass_frequency_block",
                    "hardening_parent": spec["window"].name,
                    "lower_mass_fraction": float(fraction),
                    "energy_boundary_ry": float(boundary),
                    "omega_pane": int(pane_index),
                    "tuple_part": int(tuple_index),
                }
                child_specs.append(child)
                child_fits.append({
                    "times": np.asarray(times, dtype=np.complex128),
                    "weights": np.asarray(weights, dtype=np.complex128),
                    "evidence": evidence, "budget": child_budget,
                    "residual_target": float(residual_target),
                    "direct": direct,
                })
                pane_rows.append({
                    "name": child["window"].name,
                    "kind": child["window"].kind,
                    "tuple_count": int(child["state_indices"].size),
                    "node_count": int(np.asarray(times).size),
                    "direct": bool(direct),
                    "budget_share": float(share),
                })
            pane_receipts.append(pane_rows)
            if not accepted:
                break
        tau_pairs = int(sum(fit["times"].size for fit in child_fits))
        direct_terms = int(sum(
            child["state_indices"].size
            for child, fit in zip(child_specs, child_fits) if fit["direct"]))
        trials.append({
            "lower_mass_fraction": float(fraction),
            "energy_boundary_ry": float(boundary),
            "accepted": bool(accepted),
            "tau_pairs": tau_pairs, "direct_term_count": direct_terms,
            "panes": pane_receipts, "refusal": refusal,
        })
        if accepted:
            receipt = {
                "triggered": True,
                "route": "aligned_2x2_tuple_frequency_blocks",
                "selected_lower_mass_fraction": float(fraction),
                "selected_energy_boundary_ry": float(boundary),
                "trials": trials,
            }
            for fit in child_fits:
                fit["evidence"]["hardening_receipt"] = receipt
            return child_specs, child_fits
    raise RuntimeError(
        "hybrid crossing hardening found no accepted tuple/frequency-block "
        f"plan for {int(spec['state_indices'].size)} explicit tuples: {trials}")


def _stable_time_union(fits):
    """Return one exact stable union and each free rule's union indices."""
    values, lookup, row_indices = [], {}, []
    for fit in fits:
        indices = []
        for value in np.asarray(fit["times"], np.complex128):
            key = (float(value.real), float(value.imag))
            if key not in lookup:
                lookup[key] = len(values)
                values.append(complex(value))
            indices.append(lookup[key])
        row_indices.append(np.asarray(indices, dtype=np.int64))
    return np.asarray(values, dtype=np.complex128), row_indices


def _shared_branch_grid(specs, fits, max_nodes, amp_cap):
    """Fit one common time grid with independent weights for branch windows.

    The candidate set is the exact union of already accepted incumbent-
    discipline rules. Candidates are ordered by their normalized weight across
    windows, then progressively refitted with the minimax service's fixed-time
    IRLS solver. Every trial is judged on the refined lattice and the same p99
    amplification cap as the free rules. The full union with zero-padded free
    weights is the deterministic fallback, so grid sharing cannot weaken an
    already accepted plan when that union fits under ``max_nodes``.
    """
    union, free_indices = _stable_time_union(fits)
    if union.size == 0:
        raise RuntimeError("shared tau grid received no fitted nodes")
    score = np.zeros(union.size, dtype=np.float64)
    for fit, indices in zip(fits, free_indices):
        strength = np.abs(np.asarray(fit["weights"], np.complex128))
        scale = max(float(np.sum(strength)), 1.0e-300)
        np.add.at(score, indices, strength / scale)
    order = np.lexsort((union.imag, union.real, np.abs(union), -score))
    candidates = union[order]
    inverse_order = np.empty(order.size, dtype=np.int64)
    inverse_order[order] = np.arange(order.size, dtype=np.int64)
    seeded = []
    for fit, indices in zip(fits, free_indices):
        weights = np.zeros(candidates.size, dtype=np.complex128)
        weights[inverse_order[indices]] = np.asarray(
            fit["weights"], dtype=np.complex128)
        seeded.append(weights)

    limit = min(int(max_nodes), int(candidates.size))
    coarse = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 80, 96,
              112, 128, 160, 200, 256, 384, 512, limit)
    ranks = sorted(set(rank for rank in coarse if 0 < rank <= limit))
    attempts = []
    chosen = None
    for rank in ranks:
        times = candidates[:rank]
        weights_by_window, metrics, accepted = [], [], True
        for spec, fit, seed_weights in zip(specs, fits, seeded):
            if rank == candidates.size:
                weights = seed_weights
                solver = "zero_padded_free_union"
            else:
                try:
                    weights, _objective = solve_fixed_time_weights_fast(
                        spec["problem"], times,
                        conditioning_slack=1.0e-3,
                        start_weights=seed_weights[:rank])
                    solver = "fixed_time_irls"
                except (FloatingPointError, OverflowError,
                        np.linalg.LinAlgError, ValueError):
                    accepted = False
                    break
            base = _rule_metrics(spec["problem"], times, weights)
            check = _rule_metrics(spec["validation"], times, weights)
            good = bool(
                np.all(np.isfinite(weights))
                and check[0] <= fit["residual_target"]
                and check[1] <= amp_cap)
            accepted &= good
            weights_by_window.append(weights)
            metrics.append((base, check, solver))
        attempts.append({
            "node_count": int(rank), "accepted": bool(accepted),
            "window_refined_residuals": [row[1][0] for row in metrics],
            "window_amplification_p99": [row[1][1] for row in metrics],
        })
        if accepted and len(weights_by_window) == len(fits):
            chosen = (times, weights_by_window, metrics)
            break
    if chosen is None:
        raise RuntimeError(
            "shared per-branch tau grid missed a refined target or the "
            f"amplification cap within max_nodes={int(max_nodes)}; "
            f"free-union nodes={int(candidates.size)}")

    times, weights_by_window, metrics = chosen
    for fit, weights, metric in zip(fits, weights_by_window, metrics):
        base, check, solver = metric
        fit["times"] = np.asarray(times, dtype=np.complex128)
        fit["weights"] = np.asarray(weights, dtype=np.complex128)
        fit["evidence"] = {
            **fit["evidence"],
            "free_family": fit["evidence"]["family"],
            "family": "shared_branch_grid",
            "shared_weight_solver": solver,
            "fit_residual": base[0], "refined_residual": check[0],
            "amplification_p99": check[1],
            "amplification_max": check[2],
            "shared_grid_node_count": int(times.size),
            "shared_grid_free_union_count": int(candidates.size),
            "shared_grid_attempts": attempts,
        }


def _all_pole_bounds(n_poles):
    bounds = np.asarray(
        (0.0, np.inf, -np.inf, -np.inf, np.inf, np.inf),
        dtype=np.float64)
    return np.broadcast_to(bounds, (int(n_poles), 6)).copy()


def build_delivered_sigma_windows(
    Omega_poles_by_branch,
    B_poles_by_branch,
    branches: Sequence[_SigmaBranch],
    omega_grid_ry,
    *,
    regularization_width_ry: float,
    target_error: float,
    state_amplitudes_by_branch=None,
    max_nodes: int = 512,
    lattice_bins: int = DEFAULT_LATTICE_BINS,
    amplification_cap: float = DEFAULT_AMPLIFICATION_CAP,
    true_error_safety: float = TRUE_ERROR_SAFETY,
    crossing_eps_q: float = 1.0e-3,
    use_shipped_minimax_tables: bool = True,
    pane_times: tuple = (),
    tau_grid_mode: str = "free",
    measures_by_branch=None,
):
    """Build a measure-apportioned hybrid plan for MPA or one-pole GN-PPM.

    Pole/residue collections contain one array per causal branch (or mappings
    keyed by ``branch.tag``); independent positive/negative W producers are
    never collapsed through a time-reversal assumption. The planner
    partitions explicit state--leading-pole tuples, which the MPA executor
    factors into small state-selector/pole-weight components without
    materializing a tuple axis on any spatial tensor.

    The absolute budget is ``target_error * combined inverse-gap envelope *
    true_error_safety``. The envelope bounds the scalar weighted reciprocal
    error represented by the fitting measure; it is conservative evidence,
    not a claim about cancellation in the spatially projected Sigma matrix.
    """
    started = time.perf_counter()
    branch_rows = list(branches)
    if measures_by_branch is None:
        omega_rows = _per_branch(
            Omega_poles_by_branch, branch_rows, "Omega_poles_by_branch")
        residue_rows = _per_branch(
            B_poles_by_branch, branch_rows, "B_poles_by_branch")
        amplitude_rows = _optional_per_branch(
            state_amplitudes_by_branch, branch_rows,
            "state_amplitudes_by_branch")
        measure_rows = None
    else:
        measure_rows = _per_branch(
            measures_by_branch, branch_rows, "measures_by_branch")
    omega_grid = np.asarray(omega_grid_ry, dtype=np.float64)
    eta, target = float(regularization_width_ry), float(target_error)
    amp_cap, safety = float(amplification_cap), float(true_error_safety)
    grid_mode = str(tau_grid_mode).strip().lower()
    if (omega_grid.ndim != 1 or not omega_grid.size
            or not np.all(np.isfinite(omega_grid))):
        raise ValueError("omega_grid_ry must be a nonempty finite vector")
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("delivered Sigma regularization must be finite and positive")
    if not 0.0 < target < 1.0:
        raise ValueError("delivered Sigma target_error must lie in (0,1)")
    if not 0.0 < safety <= 1.0:
        raise ValueError("true_error_safety must lie in (0,1]")
    if not np.isfinite(amp_cap) or amp_cap <= 1.0:
        raise ValueError("amplification_cap must be finite and greater than one")
    if grid_mode not in ("free", "shared"):
        raise ValueError("tau_grid_mode must be 'free' or 'shared'")
    if measure_rows is None:
        measure_rows = [
            _pole_measures(
                branch, Omega, B, eta, amplitude, int(lattice_bins))
            for branch, Omega, B, amplitude in zip(
                branch_rows, omega_rows, residue_rows, amplitude_rows)
        ]

    specs, branch_reports = [], []
    combined_envelope = np.zeros(omega_grid.size, dtype=np.float64)
    for branch, measure in zip(branch_rows, measure_rows):
        indices = np.asarray(branch.omega_idx, dtype=np.int64)
        frequencies = omega_grid[indices]
        expected = (-np.asarray(branch.omega_abs, dtype=np.float64)
                    if branch.neg_omega_half
                    else np.asarray(branch.omega_abs, dtype=np.float64))
        if not np.allclose(frequencies, expected, rtol=0.0, atol=1.0e-13):
            raise ValueError(
                f"branch {branch.tag!r} frequency indices disagree with its signed half")
        (signed_energy, state_mass, live_state_indices,
         pole_cells, pole_weights, masses, representative, state_indices,
         pole_indices, raw_count) = measure
        windows = _partition_tuples(
            representative, masses, frequencies, branch.space,
            force_single=(np.unique(pole_indices).size == 1))
        report = {
            "tag": branch.tag, "space": branch.space,
            "negative_frequency_half": bool(branch.neg_omega_half),
            "raw_tuple_count": raw_count,
            "live_tuple_count": int(np.count_nonzero(masses)),
            "live_pole_count": int(np.unique(pole_indices).size),
            "plan_start": len(specs), "windows": [],
        }
        for window in windows:
            problem, validation = _tuple_window_problems(
                window.member_indices, state_indices, pole_indices,
                live_state_indices, signed_energy, state_mass,
                pole_cells, pole_weights, frequencies,
                1.0 if branch.space == "cond" else -1.0,
                int(lattice_bins))
            envelope_by_frequency = (
                problem.cell_masses[None, :] / np.abs(problem.denominators)
            ).sum(axis=1)
            envelope = float(np.max(envelope_by_frequency))
            mass = float(np.sum(problem.cell_masses))
            kind = _window_kind(problem)
            key = f"{branch.tag}:{window.name}"
            measured_window = replace(
                window, name=key, kind=kind, delivered_mass=mass,
                mass_fraction=mass / float(np.sum(masses)))
            specs.append({
                "branch": branch, "problem": problem,
                "validation": validation, "window": measured_window,
                "member_indices": np.asarray(
                    window.member_indices, np.int64),
                "state_indices": np.asarray(
                    state_indices[window.member_indices], np.int32),
                "pole_indices": np.asarray(
                    pole_indices[window.member_indices], np.int32),
                "tuple_state_indices": state_indices,
                "tuple_pole_indices": pole_indices,
                "tuple_masses": masses,
                "tuple_representative": representative,
                "measure_context": (
                    state_indices, pole_indices, live_state_indices,
                    signed_energy, state_mass, pole_cells, pole_weights),
                "lattice_bins": int(lattice_bins),
                "omega_positions": np.arange(
                    frequencies.size, dtype=np.int64),
                "envelope": envelope, "difficulty": envelope / mass,
                "branch_report": report,
            })
            combined_envelope[indices] += envelope_by_frequency
        report["plan_stop"] = len(specs)
        branch_reports.append(report)

    combined_scale = float(np.max(combined_envelope))
    total_absolute = target * combined_scale * safety
    budgets = apportion_true_error(
        tuple(spec["window"] for spec in specs),
        np.asarray([spec["difficulty"] for spec in specs]), total_absolute)

    execution_specs, fits = [], []
    for spec, budget in zip(specs, budgets):
        branch, problem, validation = (
            spec["branch"], spec["problem"], spec["validation"])
        residual_target = budget.absolute_error_budget / spec["envelope"]
        pole_sign = 1.0 if branch.space == "cond" else -1.0
        direct = False
        if spec["window"].kind == "crossing":
            try:
                times, weights, evidence = _fit_crossing(
                    problem, validation, residual_target, pole_sign, eta,
                    int(max_nodes), amp_cap, float(crossing_eps_q),
                    bool(use_shipped_minimax_tables), tuple(pane_times))
            except RuntimeError as exc:
                if not str(exc).startswith("hybrid crossing fit missed"):
                    raise
                try:
                    child_specs, child_fits = _fit_crossing_blocks(
                        spec, budget, pole_sign=pole_sign, eta=eta,
                        max_nodes=int(max_nodes), amp_cap=amp_cap,
                        crossing_eps_q=float(crossing_eps_q),
                        use_shipped_minimax_tables=bool(
                            use_shipped_minimax_tables),
                        pane_times=tuple(pane_times))
                except RuntimeError:
                    # The final exact route is legal only for a genuinely
                    # small parent support. It is a separate execution
                    # currency and never consumes the tau ceiling.
                    n_direct = int(spec["state_indices"].size)
                    if n_direct > int(max_nodes):
                        raise
                    direct = True
                    times = np.empty(0, dtype=np.complex128)
                    weights = np.empty(0, dtype=np.complex128)
                    evidence = {
                        "family": "exact_direct_reciprocal_fallback",
                        "fit_residual": 0.0, "refined_residual": 0.0,
                        "amplification_p99": 1.0,
                        "amplification_max": 1.0,
                        "quadrature_refusal": str(exc),
                        "direct_term_count": n_direct,
                    }
                else:
                    execution_specs.extend(child_specs)
                    fits.extend(child_fits)
                    continue
        else:
            times, weights, evidence = _fit_sign_definite(
                problem, validation, residual_target, int(max_nodes), amp_cap)
        execution_specs.append(spec)
        fits.append({
            "times": np.asarray(times, dtype=np.complex128),
            "weights": np.asarray(weights, dtype=np.complex128),
            "evidence": evidence, "budget": budget,
            "residual_target": float(residual_target),
            "direct": direct,
        })

    specs = execution_specs
    offset = 0
    for report in branch_reports:
        count = sum(spec["branch_report"] is report for spec in specs)
        report["plan_start"], report["plan_stop"] = offset, offset + count
        offset += count
    if offset != len(specs):
        raise AssertionError("delivered hardening lost an execution block")

    if grid_mode == "shared":
        for report in branch_reports:
            start, stop = int(report["plan_start"]), int(report["plan_stop"])
            quadrature = [index for index in range(start, stop)
                          if not fits[index]["direct"]]
            if quadrature:
                _shared_branch_grid(
                    [specs[index] for index in quadrature],
                    [fits[index] for index in quadrature],
                    int(max_nodes), amp_cap)

    output = []
    for spec, fit in zip(specs, fits):
        branch = spec["branch"]
        times, weights, evidence = (
            fit["times"], fit["weights"], fit["evidence"])
        budget, residual_target = fit["budget"], fit["residual_target"]
        pole_sign = 1.0 if branch.space == "cond" else -1.0
        external_sign = -1 if branch.neg_omega_half else 1
        time_exec = pole_sign * np.asarray(times, np.complex128)
        alpha_exec = (np.asarray(weights, np.complex128)
                      * np.exp(-eta * time_exec))
        nodes = MinimaxNodes(
            t=jnp.asarray(time_exec, dtype=jnp.complex128),
            alpha=jnp.asarray(alpha_exec, dtype=jnp.complex128))
        state_energy = np.asarray(gather_to_host(branch.E_A), dtype=np.float64)
        state_mask = np.asarray(gather_to_host(branch.base_mask_A), dtype=bool)
        E_ref_A = float(np.min(
            state_energy.reshape(-1)[state_mask.reshape(-1)]))
        pole_indices = spec["pole_indices"]
        state_indices = spec["state_indices"]
        is_direct = bool(fit["direct"])
        window = _SigmaWindow(
            name=spec["window"].name, nodes=nodes,
            mask_A=state_mask.reshape(state_energy.shape),
            E_ref_A=E_ref_A, E_ref_B=0.0,
            omega_sign=int(pole_sign * external_sign), project="full",
            prefactor=-1.0, max_error=float(evidence["refined_residual"]),
            provenance=(
                "hybrid delivered-mass tuple window; tail-refined lattice; "
                f"true-budget share {budget.apportionment_weight:.4g}; "
                f"{evidence['family']}; amplification p99 "
                f"{evidence['amplification_p99']:.4g}"))
        output.append(SharedSigmaWindow(
            window=window, E_A=branch.E_A,
            omega_abs=np.asarray(branch.omega_abs, dtype=np.float64)[
                spec["omega_positions"]],
            omega_idx=np.asarray(branch.omega_idx, dtype=np.int64)[
                spec["omega_positions"]],
            pole_indices=pole_indices,
            bounds=_all_pole_bounds(pole_indices.size),
            phase_real=np.zeros(pole_indices.size, dtype=bool),
            band_weight=branch.band_weight,
            state_indices=state_indices,
            direct=is_direct, pole_sign=int(pole_sign),
            direct_eta_ry=eta))
        spec["branch_report"]["windows"].append({
            "name": spec["window"].name, "kind": spec["window"].kind,
            "pole_indices": pole_indices.tolist(),
            "state_indices": state_indices.tolist(),
            "delivered_mass": spec["window"].delivered_mass,
            "measured_difficulty": spec["difficulty"],
            "absolute_error_envelope": spec["envelope"],
            "absolute_error_budget": budget.absolute_error_budget,
            "budget_fraction": budget.apportionment_weight,
            "relative_residual_target": residual_target,
            "omega_positions": np.asarray(
                spec["omega_positions"], dtype=np.int64).tolist(),
            "hardening_parent": spec.get("hardening_parent"),
            "tau_grid_mode": grid_mode,
            "node_count": int(nodes.t.size),
            "direct_term_count": int(state_indices.size) if is_direct else 0,
            "execution": "direct" if is_direct else "tau",
            **evidence,
        })

    for report in branch_reports:
        report["window_count"] = int(report["plan_stop"] - report["plan_start"])
        report["window_tau_pairs"] = int(sum(
            row["node_count"] for row in report["windows"]))
        report["node_count"] = report["window_tau_pairs"]
        rows = output[report["plan_start"]:report["plan_stop"]]
        report["distinct_tau_count"] = len({
            (float(value.real), float(value.imag))
            for row in rows
            for value in np.asarray(jax.device_get(row.window.nodes.t),
                                    dtype=np.complex128)
        })
        report["direct_term_count"] = int(sum(
            row["direct_term_count"] for row in report["windows"]))
        report["window_axis"] = "state_pole_tuple"
        report["state_support"] = "explicit"
    window_tau_pairs = int(sum(row.window.n_tau for row in output))
    distinct_tau_count = int(sum(
        report["distinct_tau_count"] for report in branch_reports))
    direct_term_count = int(sum(
        report["direct_term_count"] for report in branch_reports))
    return output, {
        "plan": "delivered", "planner": "hybrid_measure_apportioned",
        "tau_grid_mode": grid_mode,
        "eta_ry": eta, "target_error": target,
        "true_error_safety": safety,
        "planned_absolute_error_budget": total_absolute,
        "combined_inverse_gap_envelope": combined_scale,
        "lattice_bins_per_axis": int(lattice_bins),
        "amplification_cap": amp_cap,
        "n_windows": len(output),
        "n_tau": window_tau_pairs,
        "window_tau_pairs": window_tau_pairs,
        "distinct_tau_count": distinct_tau_count,
        "direct_term_count": direct_term_count,
        "plan_seconds": time.perf_counter() - started,
        "branches": branch_reports,
    }


__all__ = [
    "combine_delivered_sigma_pole_measures",
    "DEFAULT_AMPLIFICATION_CAP", "DEFAULT_LATTICE_BINS",
    "TRUE_ERROR_SAFETY", "build_delivered_sigma_windows",
    "measure_delivered_sigma_pole_batch",
]

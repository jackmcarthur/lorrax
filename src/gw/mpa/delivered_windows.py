"""Measure-apportioned hybrid planning for the shared Sigma executor.

For each causal branch the physical denominator is

``d = omega - (E + Omega)`` (conduction) or
``d = omega - (E - Omega)`` (valence).

This module is the production port of DEV-80's ``hybrid_planner.py`` and
``run_study.lattice_measure``. It reduces each addressable pole shard onto
the service's tail-refined cloud-in-cell lattice, partitions the executable
leading pole axis by delivered-mass quantiles, apportions one conservative
true-error envelope by ``mass * measured inverse-gap difficulty``, then uses
the incumbent damped reciprocal fit for sign-definite windows and the
amplification-capped pane/HGL candidate fit for crossing windows.

The result is ordinary :class:`gw.mpa.sigma_windows.SharedSigmaWindow` rows.
The existing GN spatial kernel remains the only executor: no tuple mask,
second Green's function, or second tau kernel is introduced. Consequently
window membership is an exact partition of the leading pole axis; the full
state x residue measure within each selected pole set determines its fit.
The one-pole GN-PPM reduction is the degenerate one-window case.
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
    tail_refined_lattice_measure,
)


DEFAULT_LATTICE_BINS = 25
DEFAULT_AMPLIFICATION_CAP = 10.0
TRUE_ERROR_SAFETY = 0.8
WINDOW_COUNT = {"cond": 4, "val": 3}


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
    """Return signed state energies and delivered state masses."""
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
    return pole_sign * energy[live], state_mass[live]


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


def _pole_measures(branch, Omega, B, frequencies, eta, amplitude, bins):
    """Build bounded base/refined delivered measures for every live pole."""
    signed_energy, state_mass = _branch_states(branch, amplitude)
    pole_sign = 1.0 if branch.space == "cond" else -1.0
    n_poles = int(Omega.shape[0])
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

    base, refined, pole_mass, representative = [], [], [], []
    for pole_index in range(n_poles):
        values = (np.concatenate(local_values[pole_index])
                  if local_values[pole_index] else np.empty(0, np.complex128))
        masses = (np.concatenate(local_masses[pole_index])
                  if local_masses[pole_index] else np.empty(0, np.float64))
        values, masses = _gather_variable(values, masses)
        if not values.size:
            base.append(None)
            refined.append(None)
            pole_mass.append(0.0)
            representative.append(np.nan + 1.0j * np.nan)
            continue
        pole_cells, pole_weights = _compress(values, masses, bins)
        internal = (signed_energy[:, None]
                    + pole_sign * pole_cells[None, :]).reshape(-1)
        delivered = (state_mass[:, None]
                     * pole_weights[None, :]).reshape(-1)
        cells, cell_mass, refined_cells, refined_mass = (
            tail_refined_lattice_measure(
                internal, delivered, bins_per_axis=int(bins)))
        base.append(ReciprocalMeasureProblem(
            frequencies=frequencies, internal_sums=cells,
            cell_masses=cell_mass))
        refined.append(ReciprocalMeasureProblem(
            frequencies=frequencies, internal_sums=refined_cells,
            cell_masses=refined_mass))
        total = float(np.sum(delivered))
        pole_mass.append(total)
        representative.append(complex(np.sum(delivered * internal) / total))

    raw_global = int(np.asarray(all_gather_processes(
        np.asarray([raw_count], dtype=np.int64))).sum())
    return (base, refined, np.asarray(pole_mass, np.float64),
            np.asarray(representative, np.complex128), raw_global)


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


def _partition_poles(representative, masses, frequencies, space):
    live = np.nonzero(masses > 0.0)[0]
    if live.size == 0:
        raise ValueError("delivered Sigma branch has no live poles")
    if live.size == 1:
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
        # every live pole, not an invented split of that atom.
        total = float(np.sum(masses[live]))
        value = complex(np.sum(masses[live] * representative[live]) / total)
        only = _single_measure_window(0, value, total, frequencies)
        return (replace(only, name=f"{only.name}_poles",
                        member_indices=live),)
    return tuple(replace(
        window, member_indices=live[window.member_indices],
        name=f"{window.name}_poles") for window in local)


def _combine_problems(problems, indices, frequencies):
    selected = [problems[int(index)] for index in indices
                if problems[int(index)] is not None]
    if not selected:
        raise ValueError("delivered Sigma window has no measured pole support")
    return ReciprocalMeasureProblem(
        frequencies=frequencies,
        internal_sums=np.concatenate([row.internal_sums for row in selected]),
        cell_masses=np.concatenate([row.cell_masses for row in selected]))


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
):
    """Build a measure-apportioned hybrid plan for MPA or one-pole GN-PPM.

    Pole/residue collections contain one array per causal branch (or mappings
    keyed by ``branch.tag``); independent positive/negative W producers are
    never collapsed through a time-reversal assumption. The planner first
    partitions the leading pole axis, the only pole selector the incumbent
    executor can apply without a coupled tuple kernel. Each resulting fit is
    nevertheless measured on the full state x residue distribution selected
    by that pole set.

    The absolute budget is ``target_error * combined inverse-gap envelope *
    true_error_safety``. The envelope bounds the scalar weighted reciprocal
    error represented by the fitting measure; it is conservative evidence,
    not a claim about cancellation in the spatially projected Sigma matrix.
    """
    started = time.perf_counter()
    branch_rows = list(branches)
    omega_rows = _per_branch(
        Omega_poles_by_branch, branch_rows, "Omega_poles_by_branch")
    residue_rows = _per_branch(
        B_poles_by_branch, branch_rows, "B_poles_by_branch")
    amplitude_rows = _optional_per_branch(
        state_amplitudes_by_branch, branch_rows,
        "state_amplitudes_by_branch")
    omega_grid = np.asarray(omega_grid_ry, dtype=np.float64)
    eta, target = float(regularization_width_ry), float(target_error)
    amp_cap, safety = float(amplification_cap), float(true_error_safety)
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

    specs, branch_reports = [], []
    combined_envelope = np.zeros(omega_grid.size, dtype=np.float64)
    for branch, Omega, B, amplitude in zip(
            branch_rows, omega_rows, residue_rows, amplitude_rows):
        indices = np.asarray(branch.omega_idx, dtype=np.int64)
        frequencies = omega_grid[indices]
        expected = (-np.asarray(branch.omega_abs, dtype=np.float64)
                    if branch.neg_omega_half
                    else np.asarray(branch.omega_abs, dtype=np.float64))
        if not np.allclose(frequencies, expected, rtol=0.0, atol=1.0e-13):
            raise ValueError(
                f"branch {branch.tag!r} frequency indices disagree with its signed half")
        base, refined, masses, representative, raw_count = _pole_measures(
            branch, Omega, B, frequencies, eta, amplitude, int(lattice_bins))
        windows = _partition_poles(
            representative, masses, frequencies, branch.space)
        report = {
            "tag": branch.tag, "space": branch.space,
            "negative_frequency_half": bool(branch.neg_omega_half),
            "raw_tuple_count": raw_count,
            "live_pole_count": int(np.count_nonzero(masses)),
            "plan_start": len(specs), "windows": [],
        }
        for window in windows:
            problem = _combine_problems(base, window.member_indices, frequencies)
            validation = _combine_problems(
                refined, window.member_indices, frequencies)
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
                "pole_indices": np.asarray(window.member_indices, np.int32),
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

    output = []
    for spec, budget in zip(specs, budgets):
        branch, problem, validation = (
            spec["branch"], spec["problem"], spec["validation"])
        residual_target = budget.absolute_error_budget / spec["envelope"]
        pole_sign = 1.0 if branch.space == "cond" else -1.0
        if spec["window"].kind == "crossing":
            times, weights, evidence = _fit_crossing(
                problem, validation, residual_target, pole_sign, eta,
                int(max_nodes), amp_cap, float(crossing_eps_q),
                bool(use_shipped_minimax_tables), tuple(pane_times))
        else:
            times, weights, evidence = _fit_sign_definite(
                problem, validation, residual_target, int(max_nodes), amp_cap)
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
        window = _SigmaWindow(
            name=spec["window"].name, nodes=nodes,
            mask_A=state_mask.reshape(state_energy.shape),
            E_ref_A=E_ref_A, E_ref_B=0.0,
            omega_sign=int(pole_sign * external_sign), project="full",
            prefactor=-1.0, max_error=float(evidence["refined_residual"]),
            provenance=(
                "hybrid delivered-mass pole window; tail-refined lattice; "
                f"true-budget share {budget.apportionment_weight:.4g}; "
                f"{evidence['family']}; amplification p99 "
                f"{evidence['amplification_p99']:.4g}"))
        output.append(SharedSigmaWindow(
            window=window, E_A=branch.E_A,
            omega_abs=np.asarray(branch.omega_abs, dtype=np.float64),
            omega_idx=np.asarray(branch.omega_idx, dtype=np.int64),
            pole_indices=pole_indices,
            bounds=_all_pole_bounds(pole_indices.size),
            phase_real=np.zeros(pole_indices.size, dtype=bool),
            band_weight=branch.band_weight))
        spec["branch_report"]["windows"].append({
            "name": spec["window"].name, "kind": spec["window"].kind,
            "pole_indices": pole_indices.tolist(),
            "delivered_mass": spec["window"].delivered_mass,
            "measured_difficulty": spec["difficulty"],
            "absolute_error_envelope": spec["envelope"],
            "absolute_error_budget": budget.absolute_error_budget,
            "budget_fraction": budget.apportionment_weight,
            "relative_residual_target": residual_target,
            "node_count": int(nodes.t.size), **evidence,
        })

    for report in branch_reports:
        report["window_count"] = int(report["plan_stop"] - report["plan_start"])
        report["node_count"] = int(sum(
            row["node_count"] for row in report["windows"]))
    return output, {
        "plan": "delivered", "planner": "hybrid_measure_apportioned",
        "eta_ry": eta, "target_error": target,
        "true_error_safety": safety,
        "planned_absolute_error_budget": total_absolute,
        "combined_inverse_gap_envelope": combined_scale,
        "lattice_bins_per_axis": int(lattice_bins),
        "amplification_cap": amp_cap,
        "n_windows": len(output),
        "n_tau": int(sum(row.window.n_tau for row in output)),
        "plan_seconds": time.perf_counter() - started,
        "branches": branch_reports,
    }


__all__ = [
    "DEFAULT_AMPLIFICATION_CAP", "DEFAULT_LATTICE_BINS",
    "TRUE_ERROR_SAFETY", "build_delivered_sigma_windows",
]

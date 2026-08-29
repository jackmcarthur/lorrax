"""Delivered-error MPA Sigma planning on the branch's weighted measure.

For one causal branch the physical denominator is

``d = omega - (E + Omega)`` (conduction) or
``d = omega - (E - Omega)`` (valence),

where the valence ``E`` is the signed energy below the Fermi level.  The
minimax service fits ``1/d = sum_l w_l exp(i t_l d)`` directly on the
branch's weighted tuples.  This module converts that rule to the established
``_SigmaWindow`` executor convention; it does not add a second Sigma kernel.

The crossing-aware measure and equal-mass histogram are ports of
``runs/DEV/80_minimax_delivered_error_toy_20260828/tools/run_study.py``
(``crossing_aware_measure``) and ``tools/toy_models.py``
(``histogram_measure``), respectively.  The production extension is
hierarchical: each addressable pole shard is reduced to a bounded far-support
histogram before those cells are gathered, while every tuple within 2 eV of
the requested frequency segment remains raw.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from common.collectives import (all_gather_processes, gather_to_host,
                                process_rank)
from common.units import RYD_TO_EV
from gw.minimax_screening import MinimaxNodes
from gw.mpa.sigma_windows import SharedSigmaWindow
from gw.ppm_windows import _SigmaBranch, _SigmaWindow
from minimax import (ComplexTimeSearchOptions, ReciprocalMeasureProblem,
                     fit_reciprocal_measure)


DEFAULT_HISTOGRAM_CELLS = 240
DEFAULT_NEAR_BAND_RY = 2.0 / RYD_TO_EV


def _mass_quantile_groups(sort_key, masses, bin_count):
    """Partition indices into at most ``bin_count`` equal-mass groups."""
    if sort_key.size == 0:
        return []
    order = np.argsort(sort_key, kind="stable")
    sorted_mass = masses[order]
    cumulative = np.cumsum(sorted_mass)
    midpoints = cumulative - 0.5 * sorted_mass
    bins = np.minimum(
        (midpoints * int(bin_count) / cumulative[-1]).astype(int),
        int(bin_count) - 1,
    )
    return [order[bins == index] for index in range(int(bin_count))
            if np.any(bins == index)]


def histogram_measure(internal_sums, masses, *, max_cells=DEFAULT_HISTOGRAM_CELLS):
    """Compress smooth support into an equal-mass 2-D complex histogram.

    Cell locations are mass-weighted centroids on ``(Re s, Im s)`` and cell
    masses are exact sums.  The total measure is conserved to roundoff.

    Parameters
    ----------
    internal_sums : array_like, shape (n_tuple,)
        Complex internal sums ``s`` in Ry.
    masses : array_like, shape (n_tuple,)
        Nonnegative tuple masses.
    max_cells : int
        Approximate upper bound on the number of returned cells.

    Returns
    -------
    cells, cell_masses : ndarray
        Complex centroids and their nonnegative masses.
    """
    sums = np.asarray(internal_sums, dtype=np.complex128).reshape(-1)
    mass = np.asarray(masses, dtype=np.float64).reshape(-1)
    if sums.shape != mass.shape:
        raise ValueError("internal_sums and masses must have identical shapes")
    if int(max_cells) < 1:
        raise ValueError("max_cells must be positive")
    if np.any(mass < 0.0):
        raise ValueError("histogram masses must be nonnegative")
    live = mass > 0.0
    sums, mass = sums[live], mass[live]
    if not sums.size:
        return np.empty(0, np.complex128), np.empty(0, np.float64)
    if not (np.all(np.isfinite(sums)) and np.all(np.isfinite(mass))):
        raise ValueError("histogram support and masses must be finite")

    total = float(np.sum(mass))
    real_bins = max(1, int(round(np.sqrt(4.0 * int(max_cells)))))
    imag_bins = max(1, int(round(int(max_cells) / real_bins)))
    cells, cell_mass = [], []
    for real_group in _mass_quantile_groups(sums.real, mass, real_bins):
        group_sums, group_mass = sums[real_group], mass[real_group]
        for imag_group in _mass_quantile_groups(
                group_sums.imag, group_mass, imag_bins):
            member_mass = group_mass[imag_group]
            weight = float(np.sum(member_mass))
            cells.append(np.sum(member_mass * group_sums[imag_group]) / weight)
            cell_mass.append(weight)
    out_mass = np.asarray(cell_mass, dtype=np.float64)
    error = abs(float(np.sum(out_mass)) - total)
    if error > 1.0e-12 * max(total, 1.0):
        raise AssertionError(
            f"delivered Sigma histogram lost mass: {error:.3e}")
    return (np.asarray(cells, dtype=np.complex128), out_mass)


def crossing_aware_measure(internal_sums, masses, frequencies, *,
                           max_cells=DEFAULT_HISTOGRAM_CELLS,
                           near_band_ry=DEFAULT_NEAR_BAND_RY):
    """Histogram far support while retaining every near-frequency tuple.

    Nearness is the complex-plane distance to the requested real-frequency
    segment.  The default band is 2 eV, converted to Ry.
    """
    sums = np.asarray(internal_sums, dtype=np.complex128).reshape(-1)
    mass = np.asarray(masses, dtype=np.float64).reshape(-1)
    omega = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    if sums.shape != mass.shape or not omega.size:
        raise ValueError("measure arrays must match and frequencies be nonempty")
    if not (np.isfinite(near_band_ry) and float(near_band_ry) > 0.0):
        raise ValueError("near_band_ry must be finite and positive")
    gap = np.maximum(float(np.min(omega)) - sums.real,
                     np.maximum(sums.real - float(np.max(omega)), 0.0))
    near = np.hypot(np.maximum(gap, 0.0), sums.imag) < float(near_band_ry)
    far_s, far_m = histogram_measure(
        sums[~near], mass[~near], max_cells=max_cells)
    return (np.concatenate((far_s, sums[near])),
            np.concatenate((far_m, mass[near])),
            int(np.count_nonzero(near)))


def _per_branch(values, branches, name):
    if isinstance(values, Mapping):
        try:
            return [values[branch.tag] for branch in branches]
        except KeyError as exc:
            raise ValueError(f"{name} has no entry for branch {exc.args[0]!r}") from exc
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
    if values is None:
        return [None] * len(branches)
    return _per_branch(values, branches, name)


def _branch_states(branch, amplitude):
    """Return signed energies and planning masses for one branch."""
    energy = np.asarray(gather_to_host(branch.E_A), dtype=np.float64)
    mask = np.asarray(gather_to_host(branch.base_mask_A), dtype=bool)
    if energy.shape != mask.shape:
        mask = np.reshape(mask, energy.shape)
    if branch.band_weight is None:
        occupation = np.ones(energy.shape, dtype=np.float64)
    else:
        occupation = np.abs(np.asarray(
            gather_to_host(branch.band_weight), dtype=np.float64).reshape(energy.shape))
    if amplitude is None:
        state_amplitude = np.ones(energy.shape, dtype=np.float64)
    else:
        state_amplitude = np.abs(np.asarray(
            gather_to_host(amplitude), dtype=np.complex128).reshape(energy.shape))
    state_mass = occupation * state_amplitude
    live = mask & np.isfinite(energy) & np.isfinite(state_mass) & (state_mass > 0.0)
    if not np.any(live):
        raise ValueError(f"delivered Sigma branch {branch.tag!r} has no live states")
    pole_sign = 1.0 if branch.space == "cond" else -1.0
    return pole_sign * energy[live], state_mass[live]


def _local_pole_chunks(Omega, B):
    """Yield this process's nonduplicated pole shards as host arrays."""
    if tuple(Omega.shape) != tuple(B.shape) or len(Omega.shape) < 1:
        raise ValueError("per-branch pole and residue arrays must match")
    if isinstance(Omega, jax.Array) != isinstance(B, jax.Array):
        raise ValueError("pole and residue arrays must use the same storage type")
    if not isinstance(Omega, jax.Array):
        if process_rank() == 0:
            yield np.asarray(Omega), np.asarray(B)
        return
    if bool(getattr(Omega, "is_fully_replicated", False)):
        if process_rank() == 0:
            yield (np.asarray(Omega.addressable_data(0)),
                   np.asarray(B.addressable_data(0)))
        return
    shards_O, shards_B = Omega.addressable_shards, B.addressable_shards
    if len(shards_O) != len(shards_B):
        raise ValueError("pole and residue shard counts differ")
    for shard_O, shard_B in zip(shards_O, shards_B):
        if shard_O.index != shard_B.index:
            raise ValueError("pole and residue shard layouts differ")
        yield np.asarray(shard_O.data), np.asarray(shard_B.data)


def _merge_histogram(cells, masses, new_cells, new_masses, max_cells):
    if not new_cells.size:
        return cells, masses
    if cells.size:
        new_cells = np.concatenate((cells, new_cells))
        new_masses = np.concatenate((masses, new_masses))
    return histogram_measure(new_cells, new_masses, max_cells=max_cells)


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
    return (np.concatenate([value_all[r, :n] for r, n in enumerate(count)]),
            np.concatenate([mass_all[r, :n] for r, n in enumerate(count)]))


def _branch_measure(branch, Omega, B, frequencies, eta, amplitude, *,
                    max_cells, near_band_ry):
    """Build one distributed, crossing-aware branch measure."""
    signed_energy, state_mass = _branch_states(branch, amplitude)
    pole_sign = 1.0 if branch.space == "cond" else -1.0
    far_s = np.empty(0, np.complex128)
    far_m = np.empty(0, np.float64)
    near_s, near_m = [], []
    local_raw = local_bad_B = local_bad_pole = 0

    for Omega_chunk, B_chunk in _local_pole_chunks(Omega, B):
        omega_flat = np.asarray(Omega_chunk, np.complex128).reshape(-1)
        residue_flat = np.asarray(B_chunk, np.complex128).reshape(-1)
        finite_B = np.isfinite(residue_flat)
        local_bad_B += int(np.count_nonzero(~finite_B))
        live = finite_B & (np.abs(residue_flat) > 0.0)
        finite_O = np.isfinite(omega_flat)
        gamma = -omega_flat.imag
        local_bad_pole += int(np.count_nonzero(
            live & (~finite_O | (omega_flat.real <= 0.0) | (gamma < 0.0))))
        live &= finite_O & (omega_flat.real > 0.0) & (gamma >= 0.0)
        if not np.any(live):
            continue
        # The fitted object is gamma + eta.  The raw gamma is retained in
        # the executor and eta is folded into its weights exactly once below.
        pole = (omega_flat[live].real
                - 1.0j * (gamma[live] + float(eta)))
        pole_mass = np.abs(residue_flat[live])
        for energy, mass_E in zip(signed_energy, state_mass):
            sums = energy + pole_sign * pole
            masses = float(mass_E) * pole_mass
            local_raw += int(sums.size)
            gap = np.maximum(float(np.min(frequencies)) - sums.real,
                             np.maximum(sums.real - float(np.max(frequencies)), 0.0))
            near = np.hypot(np.maximum(gap, 0.0), sums.imag) < float(near_band_ry)
            if np.any(near):
                near_s.append(sums[near])
                near_m.append(masses[near])
            if np.any(~near):
                cells, cell_mass = histogram_measure(
                    sums[~near], masses[~near], max_cells=max_cells)
                far_s, far_m = _merge_histogram(
                    far_s, far_m, cells, cell_mass, max_cells)

    bad = np.asarray(all_gather_processes(np.asarray(
        [local_bad_B, local_bad_pole], dtype=np.int64))).reshape(-1, 2).sum(axis=0)
    if int(bad[0]):
        raise ValueError(
            f"delivered Sigma poles contain {int(bad[0])} nonfinite residues")
    if int(bad[1]):
        raise ValueError(
            f"delivered Sigma poles contain {int(bad[1])} unsupported live poles")

    near_local_s = (np.concatenate(near_s) if near_s
                    else np.empty(0, np.complex128))
    near_local_m = (np.concatenate(near_m) if near_m
                    else np.empty(0, np.float64))
    far_s, far_m = _gather_variable(far_s, far_m)
    near_s_all, near_m_all = _gather_variable(near_local_s, near_local_m)
    if far_s.size:
        far_s, far_m = histogram_measure(far_s, far_m, max_cells=max_cells)
    fit_s = np.concatenate((far_s, near_s_all))
    fit_m = np.concatenate((far_m, near_m_all))
    raw_count = int(np.asarray(all_gather_processes(
        np.asarray([local_raw], dtype=np.int64))).sum())
    if not fit_s.size:
        raise ValueError(f"delivered Sigma branch {branch.tag!r} has no live tuples")
    return ReciprocalMeasureProblem(
        frequencies=frequencies, internal_sums=fit_s, cell_masses=fit_m), {
            "raw_tuple_count": raw_count,
            "fit_cell_count": int(fit_s.size),
            "near_raw_tuple_count": int(near_s_all.size),
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
    histogram_cells: int = DEFAULT_HISTOGRAM_CELLS,
    near_band_ry: float = DEFAULT_NEAR_BAND_RY,
    sector_shortcut: bool = True,
    seed_times: tuple = (),
):
    """Build one delivered-error executor window per causal Sigma branch.

    Pole/residue collections contain one array per branch (or mappings keyed
    by ``branch.tag``), so positive- and negative-frequency pole sets never
    share an implicit time-reversal assumption.  A caller may pass the same
    array objects explicitly when its pole model genuinely shares them.

    Returns
    -------
    plan : list of SharedSigmaWindow
        Structurally identical to :func:`build_shared_sigma_windows` output.
        Membership is the whole branch and every live pole is selected.
    report : dict
        Node counts and delivered-error evidence per branch.
    """
    branch_rows = list(branches)
    omega_rows = _per_branch(
        Omega_poles_by_branch, branch_rows, "Omega_poles_by_branch")
    residue_rows = _per_branch(
        B_poles_by_branch, branch_rows, "B_poles_by_branch")
    amplitude_rows = _optional_per_branch(
        state_amplitudes_by_branch, branch_rows, "state_amplitudes_by_branch")
    omega_grid = np.asarray(omega_grid_ry, dtype=np.float64)
    eta = float(regularization_width_ry)
    target = float(target_error)
    if (omega_grid.ndim != 1 or not omega_grid.size
            or not np.all(np.isfinite(omega_grid))):
        raise ValueError("omega_grid_ry must be a nonempty finite vector")
    if not (np.isfinite(eta) and eta > 0.0):
        raise ValueError("delivered Sigma regularization must be finite and positive")
    if not 0.0 < target < 1.0:
        raise ValueError("delivered Sigma target_error must lie in (0, 1)")

    output, branch_reports = [], []
    for branch, Omega, B, amplitude in zip(
            branch_rows, omega_rows, residue_rows, amplitude_rows):
        indices = np.asarray(branch.omega_idx, dtype=np.int64)
        frequencies = omega_grid[indices]
        expected = (-np.asarray(branch.omega_abs, dtype=np.float64)
                    if branch.neg_omega_half
                    else np.asarray(branch.omega_abs, dtype=np.float64))
        if not np.allclose(frequencies, expected, rtol=0.0, atol=1.0e-13):
            raise ValueError(
                f"branch {branch.tag!r} frequency indices disagree with "
                "its signed half")
        problem, measure_report = _branch_measure(
            branch, Omega, B, frequencies, eta, amplitude,
            max_cells=int(histogram_cells), near_band_ry=float(near_band_ry))
        options = ComplexTimeSearchOptions(
            target_error=target, max_nodes=int(max_nodes),
            sector_shortcut=bool(sector_shortcut), seed_times=tuple(seed_times))
        rule = fit_reciprocal_measure(problem, options)
        if rule.sampled_max_error > target * (1.0 + 1.0e-9):
            raise RuntimeError(
                f"delivered Sigma branch {branch.tag!r} missed target {target:.3e}: "
                f"measured {rule.sampled_max_error:.3e}")

        pole_sign = 1.0 if branch.space == "cond" else -1.0
        external_sign = -1 if branch.neg_omega_half else 1
        # The executor stores positive hole energies and always synthesizes
        # exp[-i(E_A + Omega)t].  Valence therefore uses t_exec=-t_fit and
        # omega_sign=-sign(omega); conduction uses the fitted orientation.
        time_exec = pole_sign * np.asarray(rule.time_nodes, np.complex128)
        # eta is absent from the resident Omega tensor and enters once here.
        alpha_exec = (np.asarray(rule.weights, np.complex128)
                      * np.exp(-eta * time_exec))
        nodes = MinimaxNodes(
            t=jnp.asarray(time_exec, dtype=jnp.complex128),
            alpha=jnp.asarray(alpha_exec, dtype=jnp.complex128))
        state_energy = np.asarray(gather_to_host(branch.E_A), dtype=np.float64)
        state_mask = np.asarray(gather_to_host(branch.base_mask_A), dtype=bool)
        E_ref_A = float(np.min(state_energy.reshape(-1)[state_mask.reshape(-1)]))
        # Any reference cancels between the kernel and omega coefficient;
        # zero avoids a second global pole reduction solely for conditioning.
        window = _SigmaWindow(
            name="delivered", nodes=nodes,
            mask_A=state_mask.reshape(state_energy.shape),
            E_ref_A=E_ref_A, E_ref_B=0.0,
            omega_sign=int(pole_sign * external_sign), project="full",
            prefactor=-1.0, max_error=float(rule.sampled_max_error),
            provenance=(f"delivered-error measure; {rule.one_line()}; "
                        f"{measure_report['fit_cell_count']} fit cells, "
                        f"{measure_report['near_raw_tuple_count']} near tuples raw"))
        n_poles = int(Omega.shape[0])
        output.append(SharedSigmaWindow(
            window=window, E_A=branch.E_A,
            omega_abs=np.asarray(branch.omega_abs, dtype=np.float64),
            omega_idx=np.asarray(branch.omega_idx, dtype=np.int64),
            pole_indices=np.arange(n_poles, dtype=np.int32),
            bounds=_all_pole_bounds(n_poles),
            phase_real=np.zeros(n_poles, dtype=bool),
            band_weight=branch.band_weight))
        branch_reports.append({
            "tag": branch.tag,
            "space": branch.space,
            "negative_frequency_half": bool(branch.neg_omega_half),
            "node_count": int(rule.node_count),
            "delivered_error_max": float(rule.sampled_max_error),
            "delivered_error_by_frequency": np.asarray(
                rule.delivered_error_by_frequency, dtype=np.float64),
            "method": rule.method,
            "amplification_p99": float(rule.amplification),
            "amplification_max": float(rule.amplification_max),
            **measure_report,
        })

    return output, {
        "plan": "delivered",
        "eta_ry": eta,
        "target_error": target,
        "near_band_ry": float(near_band_ry),
        "histogram_cells": int(histogram_cells),
        "n_windows": len(output),
        "n_tau": int(sum(row.window.n_tau for row in output)),
        "branches": branch_reports,
    }


__all__ = [
    "DEFAULT_HISTOGRAM_CELLS", "DEFAULT_NEAR_BAND_RY",
    "build_delivered_sigma_windows", "crossing_aware_measure",
    "histogram_measure",
]

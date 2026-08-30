"""Synthetic gates for the delivered hybrid MPA/GN-PPM planning seam."""

import numpy as np
import jax.numpy as jnp
import pytest

from gw.mpa.delivered_windows import build_delivered_sigma_windows
from gw.ppm_accumulators import _omega_coefficient
from gw.ppm_windows import _SigmaBranch
from gw.sigma_plan import resolve_delivered_tau_grid, resolve_sigma_plan
from minimax import tail_refined_lattice_measure


def _branch(tag, space, energies, omega):
    energy = jnp.asarray(np.asarray(energies, dtype=np.float64)[None, :])
    return _SigmaBranch(
        tag, energy, jnp.ones_like(energy, dtype=bool), space, False,
        np.asarray(omega, dtype=np.float64),
        np.arange(np.asarray(omega).size, dtype=np.int64))


def _executed_scalar(plan, poles, residues, energies, omega):
    output = np.zeros(np.asarray(omega).shape, dtype=np.complex128)
    for row in plan:
        win = row.window
        if row.direct:
            external_sign = int(win.omega_sign) * int(row.pole_sign)
            for frequency, frequency_index in zip(
                    row.omega_abs, row.omega_idx):
                for state_index, pole_index in zip(
                        row.state_indices, row.pole_indices):
                    denominator = (
                        external_sign * frequency
                        - row.pole_sign * (
                            energies[state_index] + poles[pole_index]
                            - 1j * row.direct_eta_ry))
                    output[frequency_index] += (
                        win.prefactor * residues[pole_index] / denominator)
            continue
        times = np.asarray(win.nodes.t)
        alpha = np.asarray(win.nodes.alpha)
        for frequency_index, frequency in zip(row.omega_idx, row.omega_abs):
            coefficient = _omega_coefficient(
                np, frequency, times, alpha, win.omega_sign,
                win.prefactor, e_ref=win.E_ref_A + win.E_ref_B)
            for state_index, pole_index in zip(
                    row.state_indices, row.pole_indices):
                energy = energies[state_index]
                green = np.exp(-1.0j * (energy - win.E_ref_A) * times)
                screened = residues[pole_index] * np.exp(
                    -1.0j * (poles[pole_index] - win.E_ref_B) * times)
                output[frequency_index] += np.sum(
                    coefficient * green * screened)
    return output


def test_tail_refined_lattice_conserves_degenerate_one_pole_measure():
    support = np.full(7, 1.25 - 0.2j)
    mass = np.linspace(0.1, 0.7, support.size)
    cells, weights, refined, refined_weights = tail_refined_lattice_measure(
        support, mass, bins_per_axis=8)
    assert cells.shape == refined.shape == (1,)
    np.testing.assert_allclose(cells, support[:1])
    np.testing.assert_allclose(refined, support[:1])
    np.testing.assert_allclose(weights.sum(), mass.sum())
    np.testing.assert_allclose(refined_weights.sum(), mass.sum())


def test_gn_single_pole_reduction_is_one_executable_window():
    omega = np.asarray([0.2, 0.5, 0.8])
    branch = _branch("GN valence", "val", [0.3, 0.6], omega)
    poles = np.asarray([1.4 - 0.12j])
    residues = np.asarray([0.7 + 0.2j])
    plan, report = build_delivered_sigma_windows(
        [poles], [residues], [branch], omega,
        regularization_width_ry=0.04, target_error=2.0e-5,
        lattice_bins=8, max_nodes=128)
    assert report["planner"] == "hybrid_measure_apportioned"
    assert report["branches"][0]["live_pole_count"] == 1
    assert report["branches"][0]["window_count"] == 1
    assert len(plan) == 1
    np.testing.assert_array_equal(plan[0].pole_indices, [0, 0])
    np.testing.assert_array_equal(plan[0].state_indices, [0, 1])
    assert plan[0].window.project == "full"


def test_mpa_pole_windows_reconstruct_a_small_true_sigma():
    omega = np.linspace(0.0, 0.9, 7)
    energies = np.asarray([0.15, 0.55])
    branch = _branch("MPA conduction", "cond", energies, omega)
    poles = np.asarray([0.35 - 0.08j, 1.25 - 0.16j, 2.2 - 0.25j])
    residues = np.asarray([0.8 + 0.1j, -0.3 + 0.25j, 0.12 - 0.2j])
    eta = 0.05
    pane_times = tuple(np.geomspace(0.01, 160.0, 120))
    plan, report = build_delivered_sigma_windows(
        [poles], [residues], [branch], omega,
        regularization_width_ry=eta, target_error=2.0e-3,
        lattice_bins=10, max_nodes=120, pane_times=pane_times)
    membership = np.concatenate([row.pole_indices for row in plan])
    np.testing.assert_array_equal(
        np.unique(membership), np.arange(poles.size))
    assert membership.size == energies.size * poles.size
    assert report["n_windows"] >= 2
    assert all(row["amplification_p99"] <= report["amplification_cap"]
               for row in report["branches"][0]["windows"])

    executed = _executed_scalar(plan, poles, residues, energies, omega)
    broadened = poles.real - 1.0j * (-poles.imag + eta)
    denominator = (omega[:, None, None] - energies[None, :, None]
                   - broadened[None, None, :])
    direct = -np.sum(residues[None, None, :] / denominator, axis=(1, 2))
    relative = np.max(np.abs(executed - direct)) / np.max(np.abs(direct))
    assert relative <= 2.0e-3


def test_shared_grid_uses_one_branch_grid_at_matched_true_error():
    omega = np.linspace(0.0, 0.9, 7)
    energies = np.asarray([0.15, 0.55])
    branch = _branch("MPA conduction", "cond", energies, omega)
    poles = np.asarray([0.35 - 0.08j, 1.25 - 0.16j, 2.2 - 0.25j])
    residues = np.asarray([0.8 + 0.1j, -0.3 + 0.25j, 0.12 - 0.2j])
    eta = 0.05
    plan, report = build_delivered_sigma_windows(
        [poles], [residues], [branch], omega,
        regularization_width_ry=eta, target_error=2.0e-3,
        lattice_bins=10, max_nodes=120,
        pane_times=tuple(np.geomspace(0.01, 160.0, 120)),
        tau_grid_mode="shared")

    grids = [np.asarray(row.window.nodes.t) for row in plan]
    assert len(grids) >= 2
    for grid in grids[1:]:
        np.testing.assert_array_equal(grid, grids[0])
    assert report["tau_grid_mode"] == "shared"
    assert report["window_tau_pairs"] == len(plan) * grids[0].size
    assert report["distinct_tau_count"] == grids[0].size
    assert report["direct_term_count"] == 0
    assert report["branches"][0]["window_axis"] == "state_pole_tuple"
    assert report["branches"][0]["state_support"] == "explicit"

    executed = _executed_scalar(plan, poles, residues, energies, omega)
    broadened = poles.real - 1.0j * (-poles.imag + eta)
    denominator = (omega[:, None, None] - energies[None, :, None]
                   - broadened[None, None, :])
    direct = -np.sum(residues[None, None, :] / denominator, axis=(1, 2))
    relative = np.max(np.abs(executed - direct)) / np.max(np.abs(direct))
    assert relative <= 2.0e-3


def test_shared_selector_defaults_and_refuses_unknown(monkeypatch):
    from gw import ppm_sigma
    from gw.mpa import sigma as mpa_sigma

    assert ppm_sigma.resolve_sigma_plan is resolve_sigma_plan
    assert mpa_sigma.resolve_sigma_plan is resolve_sigma_plan
    monkeypatch.delenv("LORRAX_SIGMA_PLAN", raising=False)
    assert resolve_sigma_plan() == "panes"
    monkeypatch.setenv("LORRAX_SIGMA_PLAN", " delivered ")
    assert resolve_sigma_plan() == "delivered"
    monkeypatch.setenv("LORRAX_SIGMA_PLAN", "hybrid")
    with pytest.raises(ValueError, match="panes.*delivered"):
        resolve_sigma_plan()

    monkeypatch.delenv("LORRAX_DELIVERED_TAU_GRID", raising=False)
    assert resolve_delivered_tau_grid() == "free"
    monkeypatch.setenv("LORRAX_DELIVERED_TAU_GRID", " shared ")
    assert resolve_delivered_tau_grid() == "shared"
    monkeypatch.setenv("LORRAX_DELIVERED_TAU_GRID", "union")
    with pytest.raises(ValueError, match="free.*shared"):
        resolve_delivered_tau_grid()

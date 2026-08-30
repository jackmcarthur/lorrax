"""Focused gates for the envelope-relative MPA Sigma pathway."""

import numpy as np
import jax.numpy as jnp

from gw.mpa.delivered_windows import build_delivered_sigma_windows
from gw.ppm_accumulators import _omega_coefficient
from gw.ppm_windows import _SigmaBranch


def _branch(tag, space, energies, omega, *, negative_half=False):
    E_A = jnp.asarray(np.asarray(energies, dtype=np.float64)[None, :])
    mask = jnp.ones_like(E_A, dtype=bool)
    omega = np.asarray(omega, dtype=np.float64)
    return _SigmaBranch(
        tag, E_A, mask, space, negative_half, np.abs(omega),
        np.arange(omega.size, dtype=np.int64))


def test_single_term_executed_convention_reproduces_minus_residue_over_d():
    """The executor's signs, exponents, eta fold, and orientation agree."""
    omega_grid = np.asarray([0.8])
    branch = _branch("one valence term", "val", [0.6], omega_grid)
    Omega = np.asarray([1.0 - 0.2j])
    residue = np.asarray([0.8 + 0.3j])
    eta = 0.05
    plan, _ = build_delivered_sigma_windows(
        [Omega], [residue], [branch], omega_grid,
        regularization_width_ry=eta,
        envelope_relative_target=1.0e-11,
        max_nodes=512)

    row = plan[0]
    win = row.window
    times = np.asarray(win.nodes.t)
    alpha = np.asarray(win.nodes.alpha)
    # This is the scalar form consumed by DeviceOmegaAccumulator plus the
    # exact G(t) and W(t) exponent conventions of the shared tau kernel.
    coefficient = _omega_coefficient(
        np, omega_grid[0], times, alpha, win.omega_sign, win.prefactor,
        e_ref=win.E_ref_A + win.E_ref_B)
    green = np.exp(-1j * (0.6 - win.E_ref_A) * times)
    screened = residue[0] * np.exp(
        -1j * (Omega[0] - win.E_ref_B) * times)
    executed = np.sum(coefficient * green * screened)

    broadened_pole = Omega[0].real - 1j * (0.2 + eta)
    # Signed valence energy E=-0.6 and s=E-Omega.
    denominator = omega_grid[0] - (-0.6 - broadened_pole)
    expected = -residue[0] / denominator
    np.testing.assert_allclose(executed, expected, rtol=1.0e-10, atol=0.0)


def _two_branch_plan():
    omega = np.linspace(0.0, 1.2, 7)
    energies = [0.05, 0.25, 0.45, 1.4]
    branches = [
        _branch("positive conduction", "cond", energies, omega),
        _branch("positive valence", "val", energies, omega),
    ]
    pole_sets = [
        np.asarray([0.30 - 0.05j, 0.90 - 0.12j]).reshape(2, 1, 1, 1),
        np.asarray([0.45 - 0.08j, 1.25 - 0.18j]).reshape(2, 1, 1, 1),
    ]
    residue_sets = [
        np.asarray([0.70 + 0.20j, 0.25 - 0.10j]).reshape(2, 1, 1, 1),
        np.asarray([0.40 - 0.30j, 0.60 + 0.15j]).reshape(2, 1, 1, 1),
    ]
    plan, report = build_delivered_sigma_windows(
        pole_sets, residue_sets, branches, omega,
        regularization_width_ry=0.02,
        envelope_relative_target=2.0e-4,
        max_nodes=200, amplification_cap=30.0)
    return branches, pole_sets, plan, report


def test_two_branch_plan_meets_its_measure_target_and_reports_node_counts():
    branches, _pole_sets, plan, report = _two_branch_plan()
    assert len(branches) == 2
    assert report["n_windows"] == len(plan)
    assert report["n_tau"] == sum(row.window.n_tau for row in plan)
    for branch, evidence in zip(branches, report["branches"]):
        rows = plan[evidence["plan_start"]:evidence["plan_stop"]]
        assert len(rows) == evidence["window_count"]
        assert sum(row.window.n_tau for row in rows) == evidence["node_count"]
        for row, window_evidence in zip(rows, evidence["windows"]):
            assert row.window.n_tau == window_evidence["node_count"] > 0
            assert (window_evidence["refined_residual"]
                    <= window_evidence["relative_residual_target"])
            assert (window_evidence["amplification_max"]
                    <= report["amplification_cap"])
            assert row.window.project == "full"
            np.testing.assert_array_equal(row.window.mask_A, branch.base_mask_A)


def test_tr_broken_cond_and_val_pole_sets_produce_independent_plans():
    _branches, pole_sets, plan, report = _two_branch_plan()
    assert not np.array_equal(pole_sets[0], pole_sets[1])
    assert [row["tag"] for row in report["branches"]] == [
        "positive conduction", "positive valence"]
    cond = report["branches"][0]
    val = report["branches"][1]
    cond_times = np.concatenate([
        np.asarray(row.window.nodes.t)
        for row in plan[cond["plan_start"]:cond["plan_stop"]]])
    val_times = np.concatenate([
        np.asarray(row.window.nodes.t)
        for row in plan[val["plan_start"]:val["plan_stop"]]])
    assert (cond_times.shape != val_times.shape
            or not np.allclose(cond_times, val_times))


def test_selector_defaults_to_incumbent_panes(monkeypatch):
    from gw.sigma_plan import resolve_sigma_plan

    monkeypatch.delenv("LORRAX_SIGMA_PLAN", raising=False)
    assert resolve_sigma_plan() == "panes"

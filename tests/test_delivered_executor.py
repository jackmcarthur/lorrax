"""Convention and geometry gates for the delivered Sigma executor."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp

from gw.mpa.delivered_windows import build_delivered_sigma_windows
from gw.mpa.sigma import _batch_rows, _require_product_plan
from gw.ppm_windows import _SigmaBranch


def _range_row(poles):
    poles = np.asarray(poles, np.int32)
    bounds = np.column_stack((
        poles, poles + 0.5,
        np.full(poles.size, -np.inf), np.full(poles.size, -np.inf),
        np.full(poles.size, np.inf), np.full(poles.size, np.inf),
    ))
    return SimpleNamespace(
        pole_indices=poles, bounds=bounds,
        phase_real=np.zeros(poles.size, bool))


def test_product_window_ranges_keep_one_batch_width_kernel_signature():
    """Window membership changes values, never the jitted argument shapes."""
    broad = _batch_rows(_range_row((0, 1, 2)), (0, 1, 2, 3))
    narrow = _batch_rows(_range_row((2,)), (0, 1, 2, 3))
    for selected in (broad, narrow):
        pole_indices, bounds, phase_real, states = selected
        assert pole_indices.shape == phase_real.shape == (4,)
        assert bounds.shape == (4, 6)
        assert states is None
    np.testing.assert_array_equal(broad[0][:3], [0, 1, 2])
    np.testing.assert_array_equal(narrow[0][:1], [2])
    assert np.isposinf(narrow[1][1:, 0]).all()
    assert np.isneginf(narrow[1][1:, 1]).all()


def test_mpa_executor_has_one_tau_kernel_factory_and_no_delivered_factory():
    root = Path(__file__).resolve().parents[1]
    executor = (root / "src" / "gw" / "mpa" / "sigma.py").read_text()
    kernel = (root / "src" / "gw" / "ppm_tau_kernel.py").read_text()
    tree = ast.parse(executor)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "get_shared_sigma_tau_kernel"
    ]
    assert len(calls) == 1
    assert "tuple_components" not in executor + kernel
    assert "get_shared_sigma_direct_kernel" not in executor + kernel


def test_old_receipt_with_direct_work_refuses_by_path():
    """A stale pairwise receipt is rejected before executor dispatch."""
    geometry = {"direct_term_count": 1, "branches": []}
    with np.testing.assert_raises_regex(
            ValueError,
            "delivered plan receipt '/tmp/old-plan.pkl' contains direct terms"):
        _require_product_plan(
            [_range_row((0,))], geometry,
            receipt_path="/tmp/old-plan.pkl")


def test_raw_sigma_checkpoint_precedes_every_qp_stage():
    """The expensive Sigma cube must survive a later mean-field/QP failure."""
    root = Path(__file__).resolve().parents[1]
    dispatch_tree = ast.parse(
        (root / "src" / "gw" / "sigma_dispatch.py").read_text())
    finalizer = next(
        node for node in dispatch_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "finalize_dynamic_sigma")

    def named_call_lines(tree, name):
        return [
            node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name]

    write_lines = named_call_lines(finalizer, "write_sigma_omega")
    qsgw_lines = named_call_lines(finalizer, "build_qsgw_sigma_xc")
    assert len(write_lines) == 1
    assert qsgw_lines and write_lines[0] < min(qsgw_lines)

    driver_tree = ast.parse(
        (root / "src" / "gw" / "gw_jax.py").read_text())
    main = next(
        node for node in driver_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main")
    sigma_lines = named_call_lines(main, "compute_sigma_xc")
    kin_lines = named_call_lines(main, "load_kin_ion_submatrix")
    solve_lines = named_call_lines(main, "solve_qp")
    assert len(sigma_lines) == len(kin_lines) == len(solve_lines) == 1
    assert sigma_lines[0] < kin_lines[0] < solve_lines[0]


def test_planner_refuses_an_unattainable_product_window(capsys):
    E_A = jnp.asarray([[0.3]])
    branch = _SigmaBranch(
        "positive conduction", E_A, jnp.ones_like(E_A, dtype=bool),
        "cond", False, np.asarray([0.8]), np.asarray([0], np.int64))

    with np.testing.assert_raises_regex(
            RuntimeError,
            "delivered product window 'positive conduction:resonant' "
            "refused"):
        build_delivered_sigma_windows(
            [np.asarray([0.5 - 0.1j]).reshape(1, 1, 1, 1)],
            [np.asarray([0.7 + 0.2j]).reshape(1, 1, 1, 1)],
            [branch], np.asarray([0.8]), regularization_width_ry=0.05,
            envelope_relative_target=1.0e-11, max_nodes=64)

    prefix = "[delivered-planner-window] "
    lines = [line for line in capsys.readouterr().out.splitlines()
             if line.startswith(prefix)]
    assert len(lines) == 1
    record = json.loads(lines[0][len(prefix):])
    assert record["name"] == "positive conduction:resonant"
    assert record["cell_count"] > 0
    assert record["crossing_radius_ry"] > 0.0
    assert record["gamma_min_ry"] > 0.0
    assert record["A_over_eta"] > 0.0
    assert record["A_over_gamma_min"] > 0.0
    assert record["scale_span"] >= 1.0
    assert record["delivered_mass_share"] == 1.0
    assert record["apportioned_target"] > 0.0
    assert record["status"] == "refused"
    assert record["best_achieved_residual"] is not None
    assert record["best_achieved_kappa_p99"] is not None


def test_planner_integrates_crossing_in_one_product_window(monkeypatch):
    """Crossing keeps the full frequency/state support in one rectangle."""
    E_A = jnp.asarray([[-0.4, -0.2, 0.0, 0.2]])
    omega = np.asarray([0.2, 0.4, 0.6, 0.8])
    branch = _SigmaBranch(
        "positive conduction", E_A, jnp.ones_like(E_A, dtype=bool),
        "cond", False, omega, np.arange(omega.size, dtype=np.int64))

    seen = []

    def one_node_candidate(spec, eta, max_nodes, factor_growth_cap,
                           *args, **kwargs):
        del eta, max_nodes, factor_growth_cap, args, kwargs
        seen.append((spec["kind"], spec["problem"].frequencies.copy()))
        return [{
            "times": np.asarray([0.25 + 0.0j]),
            "weights": np.asarray([0.5 - 0.1j]),
            "fit_metrics": (0.0, 1.0, 1.0),
            "metrics": (0.0, 1.0, 1.0),
            "required_target": 0.0,
            "absolute_cost": 0.0,
            "factor_growth": (0.0, 0.0),
            "evidence": {
                "family": "test_product_rule",
                "candidate_tolerance": 0.0,
                "provenance": "test_fake_rule",
            },
            "attempts": [],
        }]

    monkeypatch.setattr(
        "gw.mpa.delivered_windows._candidate_rules", one_node_candidate)
    poles = np.asarray([0.5 - 0.1j]).reshape(1, 1, 1, 1)
    residues = np.asarray([0.7 + 0.2j]).reshape(1, 1, 1, 1)
    plan, report = build_delivered_sigma_windows(
        [poles], [residues], [branch], omega,
        regularization_width_ry=0.05,
        envelope_relative_target=1.0e-4,
        max_nodes=8)

    assert len(plan) == 1
    assert seen[0][0] == "crossing"
    np.testing.assert_array_equal(seen[0][1], omega)
    assert plan[0].state_indices is None
    np.testing.assert_array_equal(plan[0].window.mask_A, [[True] * 4])
    np.testing.assert_array_equal(plan[0].omega_idx, np.arange(omega.size))
    np.testing.assert_array_equal(plan[0].pole_indices, [0])
    assert report["branches"][0]["window_axis"] == (
        "state_interval_x_pole_interval")


def test_metallic_style_crossing_uses_one_bounded_product_rule():
    """A metallic crossing is integrated by one eta-damped product rule."""
    energies = np.linspace(-0.7, 0.7, 8)
    omega = np.linspace(0.1, 0.9, 5)
    E_A = jnp.asarray(energies[None, :])
    fractional_weight = jnp.asarray(
        np.linspace(0.15, 0.85, energies.size)[None, :])
    branch = _SigmaBranch(
        "recursive conduction", E_A, jnp.ones_like(E_A, dtype=bool),
        "cond", False, omega, np.arange(omega.size, dtype=np.int64),
        fractional_weight)
    poles = np.asarray([0.5 - 0.1j]).reshape(1, 1, 1, 1)
    residues = np.asarray([0.7 + 0.2j]).reshape(1, 1, 1, 1)
    plan, report = build_delivered_sigma_windows(
        [poles], [residues], [branch], omega,
        regularization_width_ry=0.05,
        envelope_relative_target=1.0e-4, max_nodes=64)

    assert len(plan) == 1
    assert report["window_tau_pairs"] <= 64
    window = report["branches"][0]["windows"][0]
    assert window["kind"] == "crossing"
    assert window["refined_residual"] <= window[
        "relative_residual_target"]
    assert window["noise_budget_met"]
    assert plan[0].state_indices is None
    np.testing.assert_array_equal(plan[0].window.mask_A, [[True] * 8])


def test_metal_oneshot_threads_one_occupation_state_to_fit_and_sigma():
    """The driver must not drop fixed-N occupations at either MPA consumer."""
    source = (Path(__file__).resolve().parents[1]
              / "src" / "gw" / "gw_jax.py").read_text()
    tree = ast.parse(source)
    calls = {"compute_screening_model": [], "compute_sigma_xc": []}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id in calls:
            calls[node.func.id].append(node)
    for name, matches in calls.items():
        assert matches, f"driver has no {name} call"
        assert any(
            keyword.arg == "occupation_state"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "oneshot_occupation_state"
            for call in matches for keyword in call.keywords
        ), f"{name} drops the one-shot metallic occupation state"

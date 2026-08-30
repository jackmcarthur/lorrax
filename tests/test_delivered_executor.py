"""Convention and geometry gates for the delivered Sigma executor."""

import ast
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from gw.minimax_screening import MinimaxNodes
from gw.mpa.delivered_windows import build_delivered_sigma_windows
from gw.mpa.sigma import (_batch_rows, _tau_groups, _tuple_components)
from gw.mpa.sigma_windows import SharedSigmaWindow
from gw.ppm_tau_kernel import (_direct_reciprocal_denominator,
                               _flat_q_difference_map)
from gw.ppm_windows import _SigmaBranch, _SigmaWindow


def _window(*, t=(0.4 + 0.1j,), alpha=(2.0 - 0.3j,), omega_idx=(0,),
            state_indices=(0,), pole_indices=(0,), direct=False,
            prefactor=-1.0, pole_sign=1, eta=0.05):
    t = np.asarray(t, np.complex128)
    alpha = np.asarray(alpha, np.complex128)
    state_indices = np.asarray(state_indices, np.int32)
    pole_indices = np.asarray(pole_indices, np.int32)
    win = _SigmaWindow(
        name="test", nodes=MinimaxNodes(jnp.asarray(t), jnp.asarray(alpha)),
        mask_A=np.ones((1, 3), bool), E_ref_A=0.0, E_ref_B=0.0,
        omega_sign=1, project="full", prefactor=prefactor)
    return SharedSigmaWindow(
        window=win, E_A=jnp.asarray([[0.2, 0.4, 0.8]]),
        omega_abs=np.asarray([0.7] * len(omega_idx)),
        omega_idx=np.asarray(omega_idx, np.int64),
        pole_indices=pole_indices,
        bounds=np.broadcast_to(
            np.asarray((0.0, np.inf, -np.inf, -np.inf, np.inf, np.inf)),
            (pole_indices.size, 6)).copy(),
        phase_real=np.zeros(pole_indices.size, bool),
        state_indices=state_indices, direct=direct,
        pole_sign=pole_sign, direct_eta_ry=eta)


def test_direct_single_term_convention_has_minus_orientation_and_one_eta():
    """The exact fallback uses -1/d and broadens the causal pole once."""
    omega, energy = 0.8, 0.6
    pole = 1.0 - 0.2j
    eta = 0.05
    denominator = _direct_reciprocal_denominator(
        omega, 1, -1, energy, pole, eta)
    executed = -1.0 / denominator
    expected = -1.0 / (omega + energy + pole.real - 1j * (0.2 + eta))
    np.testing.assert_allclose(executed, expected, rtol=5.0e-16, atol=0.0)


def test_flat_q_difference_map_is_k_minus_source_modulo_grid():
    qmap = _flat_q_difference_map((2, 2, 1))
    # source k'=(1,0,0), output k=(0,1,0) -> q=(1,1,0) -> flat 3.
    assert qmap[2, 1] == 3
    np.testing.assert_array_equal(qmap[:, 0], [0, 1, 2, 3])


def test_equal_tau_rows_fuse_once_and_keep_per_tuple_coefficients():
    row0 = _window(alpha=(2.0 - 0.3j,), state_indices=(0,),
                   pole_indices=(0,))
    row1 = _window(alpha=(-0.4 + 0.7j,), state_indices=(1,),
                   pole_indices=(1,))
    groups = _tau_groups([row0, row1], (0, 1))
    assert len(groups) == 1
    selectors, pole_weights = _tuple_components(
        groups[0], 0, np.ones((1, 3)), 2)
    assert selectors.shape == (2, 1, 3)
    np.testing.assert_array_equal(pole_weights, np.eye(2))
    # The executor folds the measured global -1 into the component before
    # the shared Sigma back-transform.
    np.testing.assert_allclose(selectors[0, 0, 0], -(2.0 - 0.3j))
    np.testing.assert_allclose(selectors[1, 0, 1], -(-0.4 + 0.7j))
    assert np.count_nonzero(selectors) == 2


def test_frequency_blocks_do_not_fuse_across_different_omega_sets():
    rows = [
        _window(omega_idx=(0,), state_indices=(0,), pole_indices=(0,)),
        _window(omega_idx=(1,), state_indices=(1,), pole_indices=(1,)),
    ]
    assert len(_tau_groups(rows, (0, 1))) == 2


def test_direct_batch_geometry_carries_ninety_one_explicit_terms():
    states = np.repeat(np.arange(13, dtype=np.int32), 7)
    poles = np.tile(np.arange(7, dtype=np.int32), 13)
    row = _window(t=(), alpha=(), state_indices=states,
                  pole_indices=poles, direct=True)
    local_poles, bounds, phase_real, got_states = _batch_rows(
        row, tuple(range(7)))
    assert local_poles.size == got_states.size == bounds.shape[0] == 91
    assert not np.any(phase_real)


def test_planner_refuses_an_unattainable_crossing_without_direct_fallback():
    E_A = jnp.asarray([[0.3]])
    branch = _SigmaBranch(
        "positive conduction", E_A, jnp.ones_like(E_A, dtype=bool),
        "cond", False, np.asarray([0.8]), np.asarray([0], np.int64))

    with np.testing.assert_raises_regex(
            RuntimeError,
            "delivered product window 'positive conduction:resonant' "
            "refused: achieved"):
        build_delivered_sigma_windows(
            [np.asarray([0.5 - 0.1j]).reshape(1, 1, 1, 1)],
            [np.asarray([0.7 + 0.2j]).reshape(1, 1, 1, 1)],
            [branch], np.asarray([0.8]), regularization_width_ry=0.05,
            envelope_relative_target=1.0e-11, max_nodes=8)


def test_planner_integrates_crossing_in_one_product_window(monkeypatch):
    """Crossing keeps the full frequency/state support in one rectangle."""
    E_A = jnp.asarray([[-0.4, -0.2, 0.0, 0.2]])
    omega = np.asarray([0.2, 0.4, 0.6, 0.8])
    branch = _SigmaBranch(
        "positive conduction", E_A, jnp.ones_like(E_A, dtype=bool),
        "cond", False, omega, np.arange(omega.size, dtype=np.int64))

    seen = []

    def one_node_candidate(spec, eta, max_nodes, factor_growth_cap):
        del eta, max_nodes, factor_growth_cap
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
    assert report["direct_term_count"] == 0
    assert plan[0].state_indices is None
    np.testing.assert_array_equal(plan[0].window.mask_A, [[True] * 4])
    np.testing.assert_array_equal(plan[0].omega_idx, np.arange(omega.size))
    np.testing.assert_array_equal(plan[0].pole_indices, [0])
    assert report["branches"][0]["window_axis"] == (
        "state_interval_x_pole_interval")


def test_metallic_style_crossing_uses_one_bounded_zero_direct_rule():
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
        envelope_relative_target=1.0e-4, max_nodes=64,
        max_direct_terms=4)

    assert len(plan) == 1
    assert report["global_direct_term_ceiling"] == 4
    assert report["direct_term_count"] == 0
    assert report["window_tau_pairs"] <= 64
    window = report["branches"][0]["windows"][0]
    assert window["kind"] == "crossing"
    assert window["refined_residual"] <= window[
        "relative_residual_target"]
    assert window["noise_budget_met"]
    assert plan[0].state_indices is None
    np.testing.assert_array_equal(plan[0].window.mask_A, [[True] * 8])
    assert not any(row.direct for row in plan)


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

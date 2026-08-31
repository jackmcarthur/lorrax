"""Focused gates for the envelope-relative MPA Sigma pathway."""

import numpy as np
import jax.numpy as jnp
import pytest

from gw.mpa import delivered_windows as delivered

from gw.mpa.delivered_windows import (
    build_delivered_sigma_windows,
    combine_delivered_sigma_pole_measures,
    load_complete_delivered_sigma_plan,
    measure_delivered_sigma_pole_batch,
)
from gw.ppm_accumulators import _omega_coefficient
from gw.ppm_windows import _SigmaBranch
from minimax import (ReciprocalMeasureProblem, delivered_error,
                     rule_amplification)


def _branch(tag, space, energies, omega, *, negative_half=False):
    E_A = jnp.asarray(np.asarray(energies, dtype=np.float64)[None, :])
    mask = jnp.ones_like(E_A, dtype=bool)
    omega = np.asarray(omega, dtype=np.float64)
    return _SigmaBranch(
        tag, E_A, mask, space, negative_half, np.abs(omega),
        np.arange(omega.size, dtype=np.int64))


def test_delivered_reduction_forwards_the_process_mesh_identity(monkeypatch):
    """The delivered planner must not construct an equal second mesh."""
    process_mesh = object()
    seen = []
    monkeypatch.setattr(delivered, "process_count", lambda: 4)
    monkeypatch.setattr(
        delivered, "psum_replicate",
        lambda local, mesh: seen.append(mesh) or np.asarray(local))

    out = delivered._sum_fixed_process_table(
        np.asarray([1, 2]), process_mesh, "identity probe")
    np.testing.assert_array_equal(out, [1, 2])
    assert seen == [process_mesh]
    assert seen[0] is process_mesh


def test_single_term_executed_convention_reproduces_minus_residue_over_d():
    """The executor's signs, exponents, eta fold, and orientation agree."""
    omega_grid = np.asarray([0.8])
    branch = _branch("one valence term", "val", [0.6], omega_grid)
    Omega = np.asarray([1.0 - 0.2j])
    residue = np.asarray([0.8 + 0.3j])
    eta = 0.05
    plan, report = build_delivered_sigma_windows(
        [Omega], [residue], [branch], omega_grid,
        regularization_width_ry=eta,
        envelope_relative_target=1.0e-5,
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
    relative_error = abs(executed - expected) / abs(expected)
    evidence = report["branches"][0]["windows"][0]
    assert relative_error <= 1.01 * evidence["refined_residual"]
    assert evidence["family"] == "noncrossing"
    assert evidence["certificate_abs_error_bound"] > 0.0
    assert "shipped noncrossing/" in evidence["fit_provenance"]


def test_streamed_pole_measures_reproduce_one_resident_measure_and_plan():
    """Pole batching changes residency, not tuple geometry or fitted rules."""
    omega_grid = np.asarray([0.0])
    branch = _branch("streamed conduction", "cond", [0.2], omega_grid)
    Omega = np.asarray([0.5 - 0.05j, 0.9 - 0.08j]).reshape(2, 1, 1, 1)
    residue = np.asarray([0.7 + 0.2j, 0.3 - 0.1j]).reshape(2, 1, 1, 1)
    kwargs = dict(regularization_width_ry=0.02, lattice_bins=9)
    whole = measure_delivered_sigma_pole_batch(
        branch, Omega, residue, **kwargs)
    streamed = combine_delivered_sigma_pole_measures([
        measure_delivered_sigma_pole_batch(
            branch, Omega[lo:lo + 1], residue[lo:lo + 1],
            pole_offset=lo, **kwargs)
        for lo in range(2)
    ])
    for index in (0, 1, 2, 5):
        np.testing.assert_allclose(streamed[index], whole[index])
    for index in (3, 4):
        for got_by_interval, expected_by_interval in zip(
                streamed[index], whole[index]):
            for got, expected in zip(got_by_interval, expected_by_interval):
                if expected is None:
                    assert got is None
                else:
                    np.testing.assert_allclose(got, expected)
    assert streamed[6] == whole[6]
    assert streamed[7] == whole[7]

    resident, resident_report = build_delivered_sigma_windows(
        [Omega], [residue], [branch], omega_grid,
        regularization_width_ry=0.02, envelope_relative_target=1.0e-5,
        lattice_bins=9, max_nodes=128)
    batched, batched_report = build_delivered_sigma_windows(
        None, None, [branch], omega_grid,
        regularization_width_ry=0.02, envelope_relative_target=1.0e-5,
        lattice_bins=9, max_nodes=128, measures_by_branch=[streamed])
    assert resident_report["window_tau_pairs"] == batched_report[
        "window_tau_pairs"]
    assert len(resident) == len(batched)
    for got, expected in zip(batched, resident):
        np.testing.assert_array_equal(got.pole_indices, expected.pole_indices)
        np.testing.assert_allclose(got.window.nodes.t, expected.window.nodes.t)
        np.testing.assert_allclose(
            got.window.nodes.alpha, expected.window.nodes.alpha)


def test_complete_plan_receipt_reconstructs_current_branch_arrays(tmp_path):
    omega_grid = np.asarray([0.0])
    branch = _branch("cached conduction", "cond", [0.2], omega_grid)
    Omega = np.asarray([0.5 - 0.05j]).reshape(1, 1, 1, 1)
    residue = np.asarray([0.7 + 0.2j]).reshape(1, 1, 1, 1)
    path = tmp_path / "delivered-plan.pkl"
    plan, report = build_delivered_sigma_windows(
        [Omega], [residue], [branch], omega_grid,
        regularization_width_ry=0.02,
        envelope_relative_target=1.0e-5,
        lattice_bins=9, max_nodes=128,
        plan_cache_path=str(path),
        plan_cache_request_fingerprint="current-input")

    loaded, loaded_report = load_complete_delivered_sigma_plan(
        str(path), "current-input", [branch])
    assert loaded_report["plan_cache_status"] == "complete_hit"
    assert loaded_report["window_tau_pairs"] == report["window_tau_pairs"]
    for got, expected in zip(loaded, plan):
        assert got.E_A is branch.E_A
        np.testing.assert_array_equal(
            got.window.mask_A, expected.window.mask_A)
        np.testing.assert_array_equal(got.pole_indices, expected.pole_indices)
        np.testing.assert_allclose(got.window.nodes.t, expected.window.nodes.t)
        np.testing.assert_allclose(
            got.window.nodes.alpha, expected.window.nodes.alpha)


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
        max_nodes=200)
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
            assert row.window.n_tau == window_evidence["node_count"]
            assert window_evidence["node_count"] > 0
            assert np.all(np.asarray(row.window.nodes.t) != 0.0)
            assert (window_evidence["refined_residual"]
                    <= window_evidence["relative_residual_target"])
            assert window_evidence["noise_budget_met"]
            assert (window_evidence["runtime_noise_bound"]
                    <= window_evidence["runtime_noise_budget"])
            assert row.window.project == "full"
            assert np.any(row.window.mask_A)
            assert np.all(np.asarray(row.window.mask_A)
                          <= np.asarray(branch.base_mask_A))
        assert 1 <= evidence["window_count"] <= 4
        assert evidence["window_axis"] == (
            "state_interval_x_pole_interval")
        assert evidence["state_support"] == "plain_interval"
    assert report["window_tau_pairs"] <= 200


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


def test_sign_definite_planning_is_lookup_only(monkeypatch):
    """A noncrossing window must never enter either optimizer."""
    def optimizer_called(*_args, **_kwargs):
        raise AssertionError("lookup-first noncrossing plan called an optimizer")

    monkeypatch.setattr(
        delivered, "solve_fixed_time_weights_fast", optimizer_called)
    branch = _branch("lookup-only valence", "val", [0.6], [0.8])
    poles = np.asarray([1.0 - 0.2j])
    residues = np.asarray([0.8 + 0.3j])
    plan, report = build_delivered_sigma_windows(
        [poles], [residues], [branch], np.asarray([0.8]),
        regularization_width_ry=0.05,
        envelope_relative_target=1.0e-5,
        max_nodes=64)

    assert len(plan) == 1
    evidence = report["branches"][0]["windows"][0]
    assert evidence["family"] == "noncrossing"
    assert evidence["catalog_achieved_abs_error"] > 0.0


def test_crossing_fallback_performs_one_fixed_time_fit(monkeypatch):
    frequencies = np.asarray([-0.5, 0.5])
    problem = ReciprocalMeasureProblem(
        frequencies=frequencies,
        internal_sums=np.asarray([0.0 - 0.05j]),
        cell_masses=np.asarray([1.0]))
    original = delivered.solve_fixed_time_weights_fast
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(delivered, "solve_fixed_time_weights_fast", counted)
    times, weights, evidence = delivered._fit_crossing_once(
        problem, pole_sign=1.0, relative_target=2.0e-2, max_nodes=64)

    assert len(calls) == 1
    assert calls[0]["iterations"] == delivered._CROSSING_FIT_ITERATIONS
    assert evidence["family"] == "crossing_fixed_time_fit"
    assert times.shape == weights.shape
    assert np.all(times != 0.0)


def _served_candidate(spec, *_args):
    evidence = {
        "family": ("crossing_hgl" if spec["kind"] == "crossing"
                   else "noncrossing"),
        "candidate_tolerance": 1.0e-6,
        "provenance": "deterministic shipped-table test rule",
    }
    return [{
        "times": np.asarray([0.1]),
        "weights": np.asarray([1.0j]),
        "evidence": evidence,
        "fit_metrics": (0.0, 1.0, 1.0),
        "metrics": (0.0, 1.0, 1.0),
        "required_target": 1.0e-5,
        "absolute_cost": spec["envelope"] * 1.0e-5,
        "factor_growth": (0.0, 0.0),
        "attempts": [],
    }]


def test_rule_selection_enforces_budget_pointwise_not_sum_of_window_maxima():
    specs = [
        {"name": "left", "envelope": 1.0,
         "omega_idx": np.asarray([0]),
         "envelope_by_frequency": np.asarray([1.0])},
        {"name": "right", "envelope": 1.0,
         "omega_idx": np.asarray([1]),
         "envelope_by_frequency": np.asarray([1.0])},
    ]
    candidates = []
    for _spec in specs:
        candidates.append([{
            "times": np.asarray([0.1]),
            "metrics": (0.08, 1.0, 1.0),
            "required_target": 0.08,
            "absolute_cost": 0.08,
        }])

    selected, nodes, required = delivered._select_rules(
        specs, candidates, total_absolute_budget=0.1, pair_ceiling=2,
        pointwise_budget=np.asarray([0.1, 0.1]))

    assert len(selected) == 2
    assert nodes == 2
    assert required == pytest.approx(0.08)
    assert sum(row["absolute_cost"] for row in selected) == pytest.approx(0.16)


def test_rule_selection_keeps_pointwise_feasible_incomparable_choice():
    specs = [
        {"name": "left", "envelope": 1.0,
         "omega_idx": np.asarray([0]),
         "envelope_by_frequency": np.asarray([1.0])},
        {"name": "right", "envelope": 1.0,
         "omega_idx": np.asarray([1]),
         "envelope_by_frequency": np.asarray([1.0])},
    ]

    def candidate(nodes, cost):
        return {
            "times": np.full(nodes, 0.1),
            "metrics": (cost, 0.0, 0.0),
            "required_target": cost,
            "absolute_cost": cost,
        }

    candidates = [
        [candidate(1, 0.9), candidate(2, 0.1)],
        [candidate(1, 0.9), candidate(2, 0.1)],
    ]

    selected, nodes, required = delivered._select_rules(
        specs, candidates, total_absolute_budget=1.0, pair_ceiling=3,
        pointwise_budget=np.asarray([1.0, 0.5]))

    assert nodes == 3
    assert [row["required_target"] for row in selected] == [0.9, 0.1]
    assert required == pytest.approx(0.9)


def _patched_test_plan(width_ev):
    ev_to_ry = 1.0 / 27.211386245988
    eta = 0.25 * ev_to_ry
    omega = np.linspace(0.0, width_ev, int(4 * width_ev) + 1) * ev_to_ry
    branch = _branch(f"{width_ev:g} eV conduction", "cond", [0.0], omega)
    poles = np.asarray([
        0.25 * width_ev * ev_to_ry,
        0.85 * width_ev * ev_to_ry,
    ], np.complex128).reshape(2, 1, 1, 1)
    residues = np.ones_like(poles)
    return build_delivered_sigma_windows(
        [poles], [residues], [branch], omega,
        regularization_width_ry=eta,
        envelope_relative_target=2.0e-4,
        lattice_bins=9, max_nodes=420)


def test_crossing_patch_count_is_the_smallest_catalog_covered_partition(
        monkeypatch):
    monkeypatch.setattr(delivered, "_candidate_rules", _served_candidate)
    narrow, narrow_report = _patched_test_plan(5.0)
    assert len(narrow) == 1
    assert narrow_report["n_windows"] == 1

    wide, wide_report = _patched_test_plan(20.0)
    crossing = [row for row in wide if ":resonant[p" in row.window.name]
    widest_span = max(
        entry.range_max
        for entry in delivered._mm.catalog().for_family("crossing")
        if entry.target_kind == "hgl"
        and entry.eps_q is not None
        and abs(entry.eps_q - delivered._CROSSING_EPS_Q) <= 1.0e-12)

    assert len(crossing) == 2
    assert all("[p" in row.window.name for row in crossing)
    assert all(
        evidence["family"] == "crossing_hgl"
        for evidence in wide_report["branches"][0]["windows"]
        if ":resonant[p" in evidence["name"])
    # One unpatched window has A=max(17, 20-5)/0.25=68, so A=60 cannot
    # cover it.  The returned two-patch cover is therefore minimal.
    assert 68.0 > widest_span


def test_emitted_crossing_patches_are_disjoint_complete_and_reproducible(
        monkeypatch):
    ev_to_ry = 1.0 / 27.211386245988
    monkeypatch.setattr(delivered, "_candidate_rules", _served_candidate)
    first, first_report = _patched_test_plan(20.0)
    second, second_report = _patched_test_plan(20.0)
    first_crossing = [
        row for row in first if ":resonant[p" in row.window.name]
    second_crossing = [
        row for row in second if ":resonant[p" in row.window.name]

    assert len(first_crossing) == len(second_crossing) == 2
    joined = np.concatenate([row.omega_idx for row in first_crossing])
    omega = np.linspace(0.0, 20.0, 81) * ev_to_ry
    np.testing.assert_array_equal(joined, np.arange(omega.size))
    assert np.unique(joined).size == joined.size
    assert all("[p" in row.window.name for row in first_crossing)
    for left, right in zip(first, second):
        assert left.window.name == right.window.name
        for left_array, right_array in (
                (left.omega_abs, right.omega_abs),
                (left.omega_idx, right.omega_idx),
                (left.window.nodes.t, right.window.nodes.t),
                (left.window.nodes.alpha, right.window.nodes.alpha),
                (left.window.mask_A, right.window.mask_A),
                (left.pole_indices, right.pole_indices),
                (left.bounds, right.bounds)):
            np.testing.assert_array_equal(left_array, right_array)
    assert (first_report["plan_cache_fingerprint"]
            == second_report["plan_cache_fingerprint"])
    assert (first_report["branches"] == second_report["branches"])


def test_zero_time_rule_is_refused():
    problem = ReciprocalMeasureProblem(
        frequencies=np.asarray([0.0]),
        internal_sums=np.asarray([-1.0 - 0.1j]),
        cell_masses=np.asarray([1.0]))
    with pytest.raises(RuntimeError, match="zero time"):
        delivered._rule_candidate(
            problem, problem, np.asarray([0.0]), np.asarray([1.0]),
            {"family": "invalid"})


def test_combined_rule_metrics_match_the_service_definitions():
    problem = ReciprocalMeasureProblem(
        frequencies=np.asarray([-0.2, 0.4]),
        internal_sums=np.asarray([-1.0 - 0.1j, -0.7 - 0.2j]),
        cell_masses=np.asarray([0.3, 0.7]))
    times = np.asarray([0.2j, 0.9j])
    weights = np.asarray([0.4 - 0.1j, 0.6 + 0.2j])
    residual, _excluded = delivered_error(problem, times, weights)
    p99, peak = rule_amplification(times, weights, problem)

    np.testing.assert_allclose(
        delivered._rule_metrics(problem, times, weights),
        (np.max(residual), p99, peak), rtol=2.0e-15, atol=0.0)


def test_noncrossing_table_walk_continues_after_a_measured_miss(monkeypatch):
    problem = ReciprocalMeasureProblem(
        frequencies=np.asarray([0.0]),
        internal_sums=np.asarray([-1.0 - 0.1j]),
        cell_masses=np.asarray([1.0]))
    denominator = complex(problem.denominators[0, 0])
    passing_time = np.asarray([1.0j])
    passing_weight = np.asarray([
        np.exp(denominator) / denominator], dtype=np.complex128)

    def table_walk(*_args, **_kwargs):
        yield (np.asarray([1.0j]), np.asarray([0.0j]),
               {"family": "noncrossing", "candidate_tolerance": 1.0e-6,
                "provenance": "first table"})
        yield (passing_time, passing_weight,
               {"family": "noncrossing", "candidate_tolerance": 2.0e-7,
                "provenance": "next tighter table"})

    monkeypatch.setattr(
        delivered, "_sign_definite_table_candidates", table_walk)
    monkeypatch.setattr(delivered, "_factor_growth", lambda *_args: (0.0, 0.0))
    spec = {
        "name": "walk probe", "kind": "sign_definite_positive",
        "problem": problem, "validation": problem, "pole_sign": 1.0,
        "envelope": 1.0,
    }
    candidates = delivered._candidate_rules(
        spec, eta=0.1, max_nodes=8, factor_growth_cap=30.0,
        relative_target=1.0e-5)

    assert len(candidates) == 1
    assert candidates[0]["evidence"]["provenance"] == "next tighter table"
    assert len(candidates[0]["attempts"]) == 2


def test_tolerance_ladder_is_deleted():
    assert not hasattr(delivered, "_FIT_TOLERANCE_LADDER")


def _emulated_parallel_runner(world):
    """Run every rank's deterministic share in-process for a CPU gate."""
    def run(count, worker, *, refuse_errors):
        shards = []
        for rank in range(world):
            rows = []
            for index in delivered._planner_work_indices(
                    count, rank=rank, world=world):
                try:
                    value, detail = worker(index)
                    error = None
                except Exception as exc:  # mirrors the production exchange
                    value, detail = None, {}
                    error = {
                        "type": type(exc).__qualname__,
                        "message": str(exc),
                    }
                rows.append({
                    "index": index, "source_rank": rank,
                    "value": value, "error": error,
                    "seconds": 0.0, "detail": detail,
                })
            shards.append(rows)
        return delivered._assemble_planner_rows(
            shards, count, world, refuse_errors=refuse_errors)
    return run


def test_parallel_planner_is_bitwise_independent_of_p(monkeypatch):
    """P=1, 4, and 16 emit identical windows, nodes, and weights."""
    def fitted(spec, *_args, **_kwargs):
        count = 5 if spec["name"].endswith(":consolidated") else 2
        offset = sum(spec["name"].encode("utf-8")) * 1.0e-6
        times = np.asarray(
            [offset + 0.1 * (index + 1) + 0.2j
             for index in range(count)], dtype=np.complex128)
        weights = np.asarray(
            [0.3j * (index + 1) - offset
             for index in range(count)], dtype=np.complex128)
        candidate = {
            "times": times, "weights": weights,
            "evidence": {
                "family": "rank-independent-test",
                "candidate_tolerance": 1.0e-6,
                "provenance": "deterministic rank partition test",
            },
            "fit_metrics": (0.0, 1.0, 1.0),
            "metrics": (0.0, 1.0, 1.0),
            "required_target": 1.0e-8,
            "absolute_cost": spec["envelope"] * 1.0e-8,
            "factor_growth": (0.0, 0.0), "attempts": [],
        }
        return [candidate], {
            "adapted_fit_seconds": 0.0,
            "shipped_fallback_seconds": 0.0,
        }

    monkeypatch.setattr(delivered, "_window_candidates_profiled", fitted)
    signatures = []
    for world in (1, 4, 16):
        monkeypatch.setattr(
            delivered, "_run_parallel_planner_jobs",
            _emulated_parallel_runner(world))
        plan, report = _patched_test_plan(20.0)
        signatures.append((
            report["n_windows"], report["window_tau_pairs"],
            tuple((
                row.window.name,
                np.asarray(row.window.nodes.t).tobytes(),
                np.asarray(row.window.nodes.alpha).tobytes(),
                np.asarray(row.window.mask_A).tobytes(),
                np.asarray(row.omega_idx).tobytes(),
                np.asarray(row.pole_indices).tobytes(),
            ) for row in plan),
        ))
    assert signatures[1] == signatures[0]
    assert signatures[2] == signatures[0]


def test_parallel_planner_refusal_is_rank_independent():
    world = 4
    shards = []
    for rank in range(world):
        rows = []
        for index in delivered._planner_work_indices(7, rank, world):
            message = None
            if index in (2, 5):
                message = f"window {index} refused on owner rank {rank}"
            rows.append({
                "index": index, "source_rank": rank, "value": index,
                "error": (None if message is None else {
                    "type": "RuntimeError", "message": message}),
                "seconds": 0.0, "detail": {},
            })
        shards.append(rows)

    for _observer_rank in range(world):
        with pytest.raises(
                RuntimeError,
                match="window 2 refused on owner rank 2"):
            delivered._assemble_planner_rows(
                shards, 7, world, refuse_errors=True)


def test_budget_retry_reuses_gathered_consolidation_trials(monkeypatch):
    """One merged fit per branch survives the global-budget retry."""
    fitted_names = []

    def fitted(spec, *_args, **_kwargs):
        fitted_names.append(spec["name"])
        return _served_candidate(spec), {
            "adapted_fit_seconds": 0.0,
            "shipped_fallback_seconds": 0.0,
        }

    original_select = delivered._select_rules
    select_calls = 0

    def retry_once(*args, **kwargs):
        nonlocal select_calls
        select_calls += 1
        if select_calls == 1:
            raise RuntimeError("forced global-budget retry")
        return original_select(*args, **kwargs)

    monkeypatch.setattr(delivered, "_window_candidates_profiled", fitted)
    monkeypatch.setattr(
        delivered, "_run_parallel_planner_jobs",
        _emulated_parallel_runner(4))
    monkeypatch.setattr(delivered, "_select_rules", retry_once)
    _branches, _poles, _plan, report = _two_branch_plan()

    assert select_calls == 2
    consolidated = [name for name in fitted_names
                    if name.endswith(":consolidated")]
    assert consolidated
    assert len(consolidated) == len(set(consolidated))
    assert report["n_windows"] > 0


def test_tightening_reaches_second_compounded_round(monkeypatch):
    """Three shortfalls reach selection call 4 and two tightened fits."""
    fitted_allowances = []

    def candidate(spec):
        row = _served_candidate(spec)[0]
        row["evidence"] = dict(
            row["evidence"], family="measure_adapted_roq")
        return row

    def fitted(spec, _eta, _ceiling, _factor_cap, allowance, **_kwargs):
        fitted_allowances.append(float(allowance))
        return [candidate(spec)], {
            "adapted_fit_seconds": 0.0,
            "shipped_fallback_seconds": 0.0,
        }

    monkeypatch.setattr(delivered, "_window_candidates_profiled", fitted)
    monkeypatch.setattr(
        delivered, "_candidate_rules",
        lambda spec, *_args, **_kwargs: [candidate(spec)])
    monkeypatch.setattr(
        delivered, "_consolidate_branches",
        lambda specs, candidates, *_args: (specs, candidates, [], None))

    select_calls = 0

    def accept_on_fourth(_specs, candidates, _budget, _ceiling):
        nonlocal select_calls
        select_calls += 1
        if select_calls < 4:
            raise delivered._BudgetShortfall("forced", 2.0, 1.0)
        selected = [rows[0] for rows in candidates]
        for row in selected:
            row["residual_target"] = row["required_target"]
        return selected, sum(row["times"].size for row in selected), 0.0

    monkeypatch.setattr(delivered, "_select_rules", accept_on_fourth)
    _plan, report = _patched_test_plan(5.0)

    assert select_calls == 4
    assert len(fitted_allowances) == 3
    np.testing.assert_allclose(
        fitted_allowances, [1.6e-4, 7.2e-5, 3.24e-5],
        rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(fitted_allowances[1:]) / fitted_allowances[0],
        [0.45, 0.45 ** 2], rtol=0.0, atol=1e-14)
    assert report["n_windows"] == 1

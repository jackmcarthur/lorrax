"""Conventions and wiring gates for the shared denominator-box plan."""

import ast
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from common.units import RYD_TO_EV
from gw.mpa.sigma import _batch_rows
from gw.ppm_windows import _SigmaBranch
from gw.sigma_box_plan import (
    _box_for_window,
    _sc_padded_box_spec,
    make_sigma_box_spec,
    plan_sigma_windows,
)
from minimax import UniformRule


def _branch(tag="positive conduction", *, space="cond", negative=False):
    omega_abs = np.asarray([0.2, 0.5], np.float64)
    return _SigmaBranch(
        tag=tag,
        E_A=jnp.asarray([[0.1, 3.0]], dtype=jnp.float64),
        base_mask_A=jnp.asarray([[True, True]]),
        space=space, neg_omega_half=negative,
        omega_abs=omega_abs,
        omega_idx=np.arange(omega_abs.size, dtype=np.int64),
    )


def _branch_at(energies, tag="positive conduction"):
    branch = _branch(tag)
    return _SigmaBranch(
        tag=branch.tag,
        E_A=jnp.asarray([energies], dtype=jnp.float64),
        base_mask_A=jnp.asarray([[True, True]]),
        space=branch.space, neg_omega_half=branch.neg_omega_half,
        omega_abs=branch.omega_abs, omega_idx=branch.omega_idx,
    )


def _summaries():
    # eta=.1, omega_max=.5, edge=1.5 -> pole edge=.65.
    shallow = (0.3, 0.3, 0.05, 0.05)
    deep = (1.0, 1.0, 0.08, 0.08)
    return (
        (0, {"all": shallow, "shallow": shallow, "deep": None}),
        (1, {"all": deep, "shallow": None, "deep": deep}),
    )


def _fake_rule(box, eps, **_kwargs):
    relative = box[0] > 0.0 or box[1] < 0.0
    return UniformRule(
        times=np.asarray([0.2 + 0.03j, 0.4 + 0.02j]),
        weights=np.asarray([0.6 - 0.1j, 0.3 + 0.05j]),
        box=tuple(box), eps=float(eps), relative=relative,
        theta_deg=5.0, rank=3, sup_error=0.5 * eps,
        kappa_max=1.2, seconds=0.01)


def _plan(monkeypatch, branch=None, **kwargs):
    monkeypatch.setattr("gw.sigma_box_plan.build_uniform_rule", _fake_rule)
    branch = _branch() if branch is None else branch
    omega = (-branch.omega_abs if branch.neg_omega_half
             else branch.omega_abs)
    return plan_sigma_windows(
        _summaries(), [branch], omega, 0.1,
        eps=1.0e-4, reduction_seconds=120.0,
        cache_dir=None, print_fn=lambda *_args, **_kwargs: None,
        **kwargs)


def test_three_product_partition_uses_raw_tuple_boxes(monkeypatch):
    plan, geometry = _plan(monkeypatch)
    assert [row.window.name for row in plan] == [
        "positive conduction:resonant",
        "positive conduction:state_tail",
        "positive conduction:pole_tail",
    ]
    assert [row.space for row in plan] == ["cond", "cond", "cond"]
    assert geometry["window_tau_pairs"] == 6
    assert geometry["eps"] == 1.0e-4
    assert geometry["rule_eps"] == 1.0e-4
    report = geometry["branches"][0]["windows"]
    assert all(row["requested_eps"] == 1.0e-4 for row in report)
    assert all(row["eps"] == 1.0e-4 for row in report)
    np.testing.assert_allclose(
        report[0]["box_ry"], [-0.206, 0.106, 0.15, 0.15])
    np.testing.assert_allclose(
        report[1]["box_ry"], [-3.106, -2.794, 0.15, 0.15])
    np.testing.assert_allclose(
        report[2]["box_ry"], [-3.864, -0.536, 0.18, 0.18])
    np.testing.assert_array_equal(plan[0].window.mask_A, [[True, False]])
    np.testing.assert_array_equal(plan[1].window.mask_A, [[False, True]])
    np.testing.assert_array_equal(plan[2].window.mask_A, [[True, True]])
    np.testing.assert_array_equal(plan[0].pole_indices, [0])
    np.testing.assert_array_equal(plan[2].pole_indices, [1])


@pytest.mark.parametrize(
    ("space", "negative"),
    (("cond", False), ("val", False), ("cond", True), ("val", True)),
)
def test_product_windows_partition_every_causal_tuple(
        monkeypatch, space, negative):
    tag = f"{'negative' if negative else 'positive'} {space}"
    branch = _branch(tag, space=space, negative=negative)
    plan, _geometry = _plan(monkeypatch, branch)
    for state in range(2):
        for pole in range(2):
            owners = [
                row.window.name
                for row in plan
                if bool(np.asarray(row.window.mask_A).reshape(-1)[state])
                and pole in set(np.asarray(row.pole_indices).tolist())
            ]
            assert len(owners) == 1, (tag, state, pole, owners)


def test_executor_conventions_and_lower_half_conjugation(monkeypatch):
    cond, _ = _plan(monkeypatch)
    fake = _fake_rule((-1.0, 1.0, 0.1, 0.2), 1.0e-4)
    first = cond[0].window
    np.testing.assert_allclose(first.nodes.t, fake.times)
    np.testing.assert_allclose(
        first.nodes.alpha, fake.weights * np.exp(-0.1 * fake.times))
    assert first.omega_sign == 1
    assert first.project == "full"
    assert first.prefactor == -1.0
    assert first.E_ref_A == 0.1 and first.E_ref_B == 0.0

    val_branch = _branch("positive valence", space="val")
    val, _ = _plan(monkeypatch, val_branch)
    # builder t,w -> -conj(t),conj(w), then executor t=pole_sign*t.
    np.testing.assert_allclose(val[0].window.nodes.t, np.conj(fake.times))
    np.testing.assert_allclose(
        val[0].window.nodes.alpha,
        np.conj(fake.weights) * np.exp(-0.1 * np.conj(fake.times)))
    assert val[0].window.omega_sign == -1


def test_containment_cache_reuses_rules_without_a_builder_call(
        monkeypatch, tmp_path):
    calls = []

    def counted(box, eps, **kwargs):
        calls.append(tuple(box))
        return _fake_rule(box, eps, **kwargs)

    monkeypatch.setattr("gw.sigma_box_plan.build_uniform_rule", counted)
    args = dict(
        eps=1.0e-4, reduction_seconds=120.0,
        cache_dir=str(tmp_path), print_fn=lambda *_args, **_kwargs: None)
    first, first_geometry = plan_sigma_windows(
        _summaries(), [_branch()], np.asarray([0.2, 0.5]), 0.1, **args)
    assert len(calls) == 3
    calls.clear()
    second, second_geometry = plan_sigma_windows(
        _summaries(), [_branch()], np.asarray([0.2, 0.5]), 0.1, **args)
    assert not calls
    assert all(window["cache_status"].startswith("hit:")
               for window in second_geometry["branches"][0]["windows"])
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left.window.nodes.t,
                                      right.window.nodes.t)
        np.testing.assert_array_equal(left.window.nodes.alpha,
                                      right.window.nodes.alpha)
    assert first_geometry["window_tau_pairs"] == second_geometry[
        "window_tau_pairs"]


def test_sc_fixed_session_reuses_identical_nodes_without_refitting(monkeypatch):
    calls = []

    def counted(box, eps, **kwargs):
        calls.append(tuple(box))
        return _fake_rule(box, eps, **kwargs)

    monkeypatch.setattr("gw.sigma_box_plan.build_uniform_rule", counted)
    session = {}
    args = dict(
        eps=1.0e-4, reduction_seconds=120.0, cache_dir=None,
        fixed_rule_session=session,
        print_fn=lambda *_args, **_kwargs: None)
    first, first_geometry = plan_sigma_windows(
        _summaries(), [_branch_at((0.1, 3.0))],
        np.asarray([0.2, 0.5]), 0.1, **args)
    assert len(calls) == 3
    calls.clear()
    second, second_geometry = plan_sigma_windows(
        _summaries(), [_branch_at((0.11, 3.01))],
        np.asarray([0.2, 0.5]), 0.1, **args)
    assert calls == []
    assert second_geometry["sc_fixed_quadrature"]
    assert not second_geometry["sc_fixed_initialized"]
    assert second_geometry["sc_fixed_rebuilds_this_iteration"] == 0
    assert second_geometry["sc_fixed_total_rebuild_count"] == 0
    assert all(row["cache_status"] == "hit:sc-fixed"
               for row in second_geometry["branches"][0]["windows"])
    first_digests = [
        row["node_digest"]
        for row in first_geometry["branches"][0]["windows"]]
    second_digests = [
        row["node_digest"]
        for row in second_geometry["branches"][0]["windows"]]
    assert first_digests == second_digests
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left.window.nodes.t,
                                      right.window.nodes.t)
        np.testing.assert_array_equal(
            left.window.nodes.alpha, right.window.nodes.alpha)
    assert first_geometry["sc_fixed_initial_window_tau_pairs"] == 6


def test_sc_fixed_session_rebuilds_an_escaped_window_and_counts_it(monkeypatch):
    calls = []

    def counted(box, eps, **kwargs):
        calls.append(tuple(box))
        return _fake_rule(box, eps, **kwargs)

    monkeypatch.setattr("gw.sigma_box_plan.build_uniform_rule", counted)
    session = {}
    args = dict(
        eps=1.0e-4, reduction_seconds=120.0, cache_dir=None,
        fixed_rule_session=session,
        print_fn=lambda *_args, **_kwargs: None)
    plan_sigma_windows(
        _summaries(), [_branch_at((0.1, 3.0))],
        np.asarray([0.2, 0.5]), 0.1, **args)
    calls.clear()
    _second, geometry = plan_sigma_windows(
        _summaries(), [_branch_at((0.1, 8.0))],
        np.asarray([0.2, 0.5]), 0.1, **args)
    # Only the escaped window(s) are refitted; the rest reuse their nodes.
    rebuilt = geometry["sc_fixed_rebuilds_this_iteration"]
    assert 1 <= rebuilt <= 3
    assert len(calls) == rebuilt
    assert geometry["sc_fixed_total_rebuild_count"] == rebuilt
    assert len(geometry["sc_fixed_rebuilt_windows"]) == rebuilt
    statuses = [row["cache_status"]
                for row in geometry["branches"][0]["windows"]]
    assert statuses.count("rebuild:sc-fixed") == rebuilt
    assert all(status in ("rebuild:sc-fixed", "hit:sc-fixed")
               for status in statuses)
    # The rebuilt certificate holds on the next map without another fit.
    calls.clear()
    _third, geometry = plan_sigma_windows(
        _summaries(), [_branch_at((0.1, 8.0))],
        np.asarray([0.2, 0.5]), 0.1, **args)
    assert calls == []
    assert geometry["sc_fixed_rebuilds_this_iteration"] == 0
    assert geometry["sc_fixed_total_rebuild_count"] == rebuilt


def test_sc_fixed_session_keeps_receipt_for_temporarily_empty_window(
        monkeypatch):
    calls = []

    def counted(box, eps, **kwargs):
        calls.append(tuple(box))
        return _fake_rule(box, eps, **kwargs)

    monkeypatch.setattr("gw.sigma_box_plan.build_uniform_rule", counted)
    session = {}
    args = dict(
        eps=1.0e-4, reduction_seconds=120.0, cache_dir=None,
        fixed_rule_session=session,
        print_fn=lambda *_args, **_kwargs: None)
    _, first_geometry = plan_sigma_windows(
        _summaries(), [_branch_at((0.1, 3.0))],
        np.asarray([0.2, 0.5]), 0.1, **args)
    calls.clear()
    _, subset_geometry = plan_sigma_windows(
        _summaries(), [_branch_at((0.1, 0.2))],
        np.asarray([0.2, 0.5]), 0.1, **args)
    assert calls == []
    _, restored_geometry = plan_sigma_windows(
        _summaries(), [_branch_at((0.1, 3.0))],
        np.asarray([0.2, 0.5]), 0.1, **args)
    assert calls == []
    assert [row["name"] for row in first_geometry["branches"][0]["windows"]] == [
        "positive conduction:resonant",
        "positive conduction:state_tail",
        "positive conduction:pole_tail",
    ]
    assert [row["name"] for row in subset_geometry["branches"][0]["windows"]] == [
        "positive conduction:resonant",
        "positive conduction:pole_tail",
    ]
    assert set(session["rules"]) == {
        "positive conduction:resonant",
        "positive conduction:state_tail",
        "positive conduction:pole_tail",
    }
    first = first_geometry["branches"][0]["windows"]
    restored = restored_geometry["branches"][0]["windows"]
    assert [row["node_digest"] for row in restored] == [
        row["node_digest"] for row in first]


def test_sc_fixed_session_rebuilds_a_window_absent_from_iteration_one(
        monkeypatch):
    calls = []

    def counted(box, eps, **kwargs):
        calls.append(tuple(box))
        return _fake_rule(box, eps, **kwargs)

    monkeypatch.setattr("gw.sigma_box_plan.build_uniform_rule", counted)
    session = {}
    args = dict(
        eps=1.0e-4, reduction_seconds=120.0, cache_dir=None,
        fixed_rule_session=session,
        print_fn=lambda *_args, **_kwargs: None)
    _first, first_geometry = plan_sigma_windows(
        _summaries(), [_branch_at((0.1, 0.2))],
        np.asarray([0.2, 0.5]), 0.1, **args)
    n_first = len(first_geometry["branches"][0]["windows"])
    calls.clear()
    _second, geometry = plan_sigma_windows(
        _summaries(), [_branch_at((0.1, 3.0))],
        np.asarray([0.2, 0.5]), 0.1, **args)
    n_second = len(geometry["branches"][0]["windows"])
    assert n_second > n_first
    rebuilt = geometry["sc_fixed_rebuilds_this_iteration"]
    assert rebuilt >= n_second - n_first
    assert len(calls) == rebuilt
    assert geometry["sc_fixed_total_rebuild_count"] == rebuilt


def test_sc_rule_padding_is_two_ev_on_state_edges_and_ten_percent_on_poles():
    eta = 0.1
    spec = make_sigma_box_spec(
        name="crossing", frequencies=(-2.0, 2.0), states=(-0.2, 0.2),
        pole_stats=((1.0, 2.0, 0.5, 1.0),), pole_sign=1.0,
        eta_ry=eta)
    assert spec["kind"] == "crossing"
    padded = _sc_padded_box_spec(spec, eta)
    expanded_poles = ((0.9, 2.2, 0.45, 1.1),)
    pole_box, _, _ = _box_for_window(
        spec["frequencies"], spec["states"], expanded_poles,
        spec["pole_sign"], eta)
    expected = (
        min(spec["box"][0], pole_box[0]) - 2.0 / RYD_TO_EV,
        max(spec["box"][1], pole_box[1]) + 2.0 / RYD_TO_EV,
        min(spec["box"][2], pole_box[2]),
        max(spec["box"][3], pole_box[3]),
    )
    np.testing.assert_allclose(padded["box"], expected, rtol=0.0, atol=0.0)
    assert padded["sc_state_pad_ev"] == 2.0
    assert padded["sc_pole_pad_fraction"] == 0.10


def test_fixed_sc_accepts_the_box_services_finite_fallback(monkeypatch):
    import dataclasses

    def diagnostic_above_eps(box, eps, **kwargs):
        return dataclasses.replace(
            _fake_rule(box, eps, **kwargs), sup_error=5.5 * eps)

    monkeypatch.setattr(
        "gw.sigma_box_plan.build_uniform_rule", diagnostic_above_eps)
    plan, geometry = plan_sigma_windows(
        _summaries(), [_branch()], np.asarray([0.2, 0.5]), 0.1,
        eps=1.0e-4, reduction_seconds=120.0,
        cache_dir=None, fixed_rule_session={},
        print_fn=lambda *_args, **_kwargs: None)
    assert len(plan) == 3
    assert all(row["sup_error"] == pytest.approx(5.5e-4)
               for row in geometry["branches"][0]["windows"])


def test_one_shot_preserves_the_historical_sup_error_refusal(monkeypatch):
    import dataclasses

    def diagnostic_above_eps(box, eps, **kwargs):
        return dataclasses.replace(
            _fake_rule(box, eps, **kwargs), sup_error=5.5 * eps)

    monkeypatch.setattr(
        "gw.sigma_box_plan.build_uniform_rule", diagnostic_above_eps)
    with pytest.raises(RuntimeError, match="rule sup error"):
        plan_sigma_windows(
            _summaries(), [_branch()], np.asarray([0.2, 0.5]), 0.1,
            eps=1.0e-4, reduction_seconds=120.0,
            cache_dir=None, print_fn=lambda *_args, **_kwargs: None)


def test_crossing_noise_gate_uses_peak_relative_term_mass(monkeypatch):
    import dataclasses

    def large_relative_kappa(box, eps, **kwargs):
        # A far crossing edge can have large cancellation RELATIVE to Q while
        # its term mass remains small relative to the 1/eta peak.
        return dataclasses.replace(
            _fake_rule(box, eps, **kwargs), kappa_max=1.0e6)

    monkeypatch.setattr(
        "gw.sigma_box_plan.build_uniform_rule", large_relative_kappa)
    plan, geometry = plan_sigma_windows(
        _summaries(), [_branch()], np.asarray([0.2, 0.5]), 0.1,
        eps=1.0e-4, reduction_seconds=120.0,
        cache_dir=None, print_fn=lambda *_args, **_kwargs: None)
    assert len(plan) == 3
    assert geometry["branches"][0]["windows"][0]["kappa_max"] == 1.0e6
    assert all(
        row["runtime_noise_bound"] <= row["runtime_noise_budget"]
        for row in geometry["branches"][0]["windows"])
    assert all(
        row["runtime_noise_budget"] == pytest.approx(5.0e-6)
        for row in geometry["branches"][0]["windows"])


def test_executor_noise_gate_refuses_large_term_mass(monkeypatch):
    import dataclasses

    def unstable(box, eps, **kwargs):
        # Opposite O(1e5) terms inflate the cancellation ratio in the rule's
        # own error currency while leaving the mocked certificate unchanged.
        return dataclasses.replace(
            _fake_rule(box, eps, **kwargs),
            weights=np.asarray([1.0e5 + 0.0j, -1.0e5 + 0.0j]))

    monkeypatch.setattr("gw.sigma_box_plan.build_uniform_rule", unstable)
    with pytest.raises(RuntimeError, match="runtime-noise"):
        plan_sigma_windows(
            _summaries(), [_branch()], np.asarray([0.2, 0.5]), 0.1,
            eps=1.0e-4, reduction_seconds=120.0,
            cache_dir=None, print_fn=lambda *_args, **_kwargs: None)


def test_sign_definite_builder_receives_executor_noise_cap(monkeypatch):
    calls = []

    def conditioned(box, eps, **kwargs):
        calls.append((tuple(box), dict(kwargs)))
        return _fake_rule(box, eps, **kwargs)

    monkeypatch.setattr("gw.sigma_box_plan.build_uniform_rule", conditioned)
    plan_sigma_windows(
        _summaries(), [_branch()], np.asarray([0.2, 0.5]), 0.1,
        eps=1.0e-4, reduction_seconds=120.0,
        cache_dir=None, print_fn=lambda *_args, **_kwargs: None)
    expected = (0.05 * 1.0e-4) / (6.0e-8 * (1.0 + 1.0e-4))
    crossing = [kwargs for box, kwargs in calls if box[0] <= 0.0 <= box[1]]
    sign_definite = [
        kwargs for box, kwargs in calls if box[0] > 0.0 or box[1] < 0.0
    ]
    assert crossing and all("kappa_cap" not in kwargs for kwargs in crossing)
    assert sign_definite and all(
        kwargs["kappa_cap"] == pytest.approx(expected)
        for kwargs in sign_definite)


def test_cache_rule_missing_active_noise_cap_does_not_shadow_builder(
        monkeypatch, tmp_path):
    monkeypatch.setattr("gw.sigma_box_plan.build_uniform_rule", _fake_rule)
    args = dict(
        eps=1.0e-4, reduction_seconds=120.0,
        cache_dir=str(tmp_path), print_fn=lambda *_args, **_kwargs: None)
    plan_sigma_windows(
        _summaries(), [_branch()], np.asarray([0.2, 0.5]), 0.1, **args)
    paths = list(tmp_path.glob("*.npz"))
    assert len(paths) == 3
    for path in paths:
        with np.load(path) as data:
            payload = {name: np.asarray(data[name]) for name in data.files}
        payload["roundoff_amplification"] = np.asarray(1.0e6)
        np.savez(path, **payload)

    calls = []

    def counted(box, eps, **kwargs):
        calls.append(tuple(box))
        return _fake_rule(box, eps, **kwargs)

    monkeypatch.setattr("gw.sigma_box_plan.build_uniform_rule", counted)
    _plan, geometry = plan_sigma_windows(
        _summaries(), [_branch()], np.asarray([0.2, 0.5]), 0.1, **args)
    assert len(calls) == 3
    assert all(
        row["cache_status"] == "miss"
        for row in geometry["branches"][0]["windows"])


def test_each_cache_write_failure_is_announced_without_rejecting_rule(
        monkeypatch, tmp_path):
    monkeypatch.setattr("gw.sigma_box_plan.build_uniform_rule", _fake_rule)
    monkeypatch.setattr(
        "gw.sigma_box_plan.os.replace",
        lambda *_args: (_ for _ in ()).throw(OSError("read-only cache")))
    lines = []
    plan, geometry = plan_sigma_windows(
        _summaries(), [_branch()], np.asarray([0.2, 0.5]), 0.1,
        eps=1.0e-4, reduction_seconds=120.0,
        cache_dir=str(tmp_path), print_fn=lines.append)

    assert len(plan) == 3
    warnings = [line for line in lines
                if line.startswith("WARNING sigma quadrature cache")]
    assert len(warnings) == 3
    assert all(str(tmp_path) in line for line in warnings)
    assert all("OSError: read-only cache" in line for line in warnings)
    assert all(row["cache_status"] == "miss"
               for row in geometry["branches"][0]["windows"])
    assert not list(tmp_path.glob("*.tmp"))


def test_no_pair_ceiling(monkeypatch):
    # Owner ruling 2026-09-02: the plan reports its pair count, never refuses on it.
    monkeypatch.setattr("gw.sigma_box_plan.build_uniform_rule", _fake_rule)
    plan, geometry = plan_sigma_windows(
        _summaries(), [_branch()], np.asarray([0.2, 0.5]), 0.1,
        eps=1.0e-4, reduction_seconds=120.0,
        cache_dir=None, print_fn=lambda *_args, **_kwargs: None)
    assert "pair_ceiling" not in geometry and geometry["window_tau_pairs"] > 5


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


def test_mpa_executor_has_one_tau_kernel_factory():
    root = Path(__file__).resolve().parents[1]
    executor = (root / "src" / "gw" / "mpa" / "sigma.py").read_text()
    tree = ast.parse(executor)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "get_shared_sigma_tau_kernel"
    ]
    assert len(calls) == 1


def test_box_route_is_default_and_campaign_route_refuses(monkeypatch):
    from gw.sigma_plan import resolve_sigma_plan

    monkeypatch.delenv("LORRAX_SIGMA_PLAN", raising=False)
    assert resolve_sigma_plan() == "box"
    monkeypatch.setenv("LORRAX_SIGMA_PLAN", "panes")
    assert resolve_sigma_plan() == "panes"
    monkeypatch.setenv("LORRAX_SIGMA_PLAN", "delivered")
    with pytest.raises(ValueError, match="box.*panes"):
        resolve_sigma_plan()


_DECK = """\
[cohsex]
nval = 2
ncond = 2
nband = 10
memory_per_device_gb = 4.0
"""


def test_quadrature_deck_defaults_and_retired_sector_key(tmp_path):
    from gw.gw_config import LorraxConfig

    deck = tmp_path / "box.in"
    deck.write_text(_DECK)
    config = LorraxConfig.from_input_file(
        str(deck), print_fn=lambda *_args, **_kwargs: None)
    assert config.sigma.quadrature_eps == 1.0e-4
    assert config.sigma.quadrature_reduction_seconds == 120.0
    assert config.sigma.quadrature_cache_dir == "auto"

    deck.write_text(
        _DECK
        + "sigma_quadrature_eps = 2e-4\n"
        + "sigma_quadrature_reduction_seconds = 30\n"
        + "sigma_quadrature_cache_dir = off\n")
    config = LorraxConfig.from_input_file(
        str(deck), print_fn=lambda *_args, **_kwargs: None)
    assert config.sigma.quadrature_eps == 2.0e-4
    assert config.sigma.quadrature_reduction_seconds == 30.0
    assert config.sigma.quadrature_cache_dir == "off"

    deck.write_text(_DECK + "mpa_sigma_sector_target_error = 1e-4\n")
    with pytest.raises(ValueError, match="retired.*sigma_quadrature_eps"):
        LorraxConfig.from_input_file(
            str(deck), print_fn=lambda *_args, **_kwargs: None)

    for key in ("ppm_sigma_target_error", "ppm_sigma_max_nodes"):
        deck.write_text(_DECK + f"{key} = 1e-4\n")
        with pytest.raises(ValueError, match=f"{key}.*retired"):
            LorraxConfig.from_input_file(
                str(deck), print_fn=lambda *_args, **_kwargs: None)

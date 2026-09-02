"""Conventions and wiring gates for the MPA denominator-box plan."""

import ast
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gw.mpa.sigma import _batch_rows
from gw.ppm_windows import _SigmaBranch
from gw.sigma_box_plan import plan_sigma_windows
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


def _pole_batches():
    # eta=.1, omega_max=.5, edge=1.5 -> pole edge=.65.  The tiny-residue
    # third pole is beyond both 99th percentiles and becomes an outlier.
    Omega = jnp.asarray([
        [[[0.3 - 0.05j]]],
        [[[1.0 - 0.08j]]],
        [[[9.0 - 3.0j]]],
    ], dtype=jnp.complex128)
    B = jnp.asarray([
        [[[100.0]]], [[[100.0]]], [[[0.1]]],
    ], dtype=jnp.complex128)

    def batches():
        yield 0, Omega, B

    return batches


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
    monkeypatch.setattr(
        "gw.sigma_box_plan.rule_amplification_p99",
        lambda *_args, **_kwargs: 1.1)
    branch = _branch() if branch is None else branch
    omega = (-branch.omega_abs if branch.neg_omega_half
             else branch.omega_abs)
    return plan_sigma_windows(
        _pole_batches(), [branch], omega, 0.1,
        eps=1.0e-4, reduction_seconds=120.0, pair_ceiling=20,
        cache_dir=None, print_fn=lambda *_args, **_kwargs: None,
        **kwargs)


def test_three_product_partition_uses_raw_tuple_boxes(monkeypatch):
    plan, geometry = _plan(monkeypatch)
    assert {row.space for row in plan} == {"cond"}
    assert [row.window.name for row in plan] == [
        "positive conduction:resonant",
        "positive conduction:state_tail",
        "positive conduction:pole_tail",
        "positive conduction:outlier-negative",
    ]
    assert geometry["window_tau_pairs"] == 8
    report = geometry["branches"][0]["windows"]
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
    np.testing.assert_array_equal(plan[3].pole_indices, [2])
    assert geometry["pole_partition"]["weight"] == "abs(B_p)"
    assert geometry["pole_partition"]["tail_fraction_per_side"] == 0.01

    # The products cover every state/pole tuple exactly once.  The outlier
    # union changes ownership, never eligibility.
    coordinates = ((0.3, 0.05), (1.0, 0.08), (9.0, 3.0))
    for state in range(2):
        for pole, (a, gamma) in enumerate(coordinates):
            owners = 0
            for row in plan:
                if not np.asarray(row.window.mask_A).reshape(-1)[state]:
                    continue
                positions = np.nonzero(row.pole_indices == pole)[0]
                for position in positions:
                    regions = np.asarray(row.bounds[position])
                    selected = (
                        (a > regions[:, 0]) & (a <= regions[:, 1])
                        & (gamma >= regions[:, 2])
                        & (gamma > regions[:, 3])
                        & (gamma < regions[:, 4])
                        & (gamma <= regions[:, 5]))
                    owners += int(np.any(selected))
            assert owners == 1


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
    assert {row.space for row in val} == {"val"}
    # builder t,w -> -conj(t),conj(w), then executor t=pole_sign*t.
    np.testing.assert_allclose(val[0].window.nodes.t, np.conj(fake.times))
    np.testing.assert_allclose(
        val[0].window.nodes.alpha,
        np.conj(fake.weights) * np.exp(-0.1 * np.conj(fake.times)))
    assert val[0].window.omega_sign == -1


def test_ppm_compatibility_broadens_only_crossing_boxes(monkeypatch):
    plan, geometry = _plan(monkeypatch, broaden_sign_definite=False)
    crossing, tail = plan[:2]
    fake_crossing = _fake_rule(
        geometry["branches"][0]["windows"][0]["box_ry"], 1.0e-4)
    fake_tail = _fake_rule(
        geometry["branches"][0]["windows"][1]["box_ry"], 1.0e-4)
    tail_eta = geometry["branches"][0]["windows"][1][
        "external_regularization_ry"]

    assert geometry["broaden_sign_definite"] is False
    assert geometry["branches"][0]["windows"][0][
        "external_regularization_ry"] == 0.1
    assert 0.0 < tail_eta < 1.0e-6
    np.testing.assert_allclose(
        crossing.window.nodes.alpha,
        fake_crossing.weights * np.exp(-0.1 * fake_crossing.times))
    np.testing.assert_allclose(
        tail.window.nodes.alpha,
        fake_tail.weights * np.exp(-tail_eta * fake_tail.times))


def test_containment_cache_reuses_rules_without_a_builder_call(
        monkeypatch, tmp_path):
    calls = []

    def counted(box, eps, **kwargs):
        calls.append(tuple(box))
        return _fake_rule(box, eps, **kwargs)

    monkeypatch.setattr("gw.sigma_box_plan.build_uniform_rule", counted)
    monkeypatch.setattr(
        "gw.sigma_box_plan.rule_amplification_p99",
        lambda *_args, **_kwargs: 1.1)
    args = dict(
        eps=1.0e-4, reduction_seconds=120.0, pair_ceiling=20,
        cache_dir=str(tmp_path), print_fn=lambda *_args, **_kwargs: None)
    first, first_geometry = plan_sigma_windows(
        _pole_batches(), [_branch()], np.asarray([0.2, 0.5]), 0.1, **args)
    assert len(calls) == 4
    calls.clear()
    second, second_geometry = plan_sigma_windows(
        _pole_batches(), [_branch()], np.asarray([0.2, 0.5]), 0.1, **args)
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


def test_total_pair_ceiling_refuses(monkeypatch):
    monkeypatch.setattr("gw.sigma_box_plan.build_uniform_rule", _fake_rule)
    monkeypatch.setattr(
        "gw.sigma_box_plan.rule_amplification_p99",
        lambda *_args, **_kwargs: 1.1)
    with pytest.raises(RuntimeError, match="pair ceiling=7"):
        plan_sigma_windows(
            _pole_batches(), [_branch()], np.asarray([0.2, 0.5]), 0.1,
            eps=1.0e-4, reduction_seconds=120.0, pair_ceiling=7,
            cache_dir=None, print_fn=lambda *_args, **_kwargs: None)


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


def _union_row(poles):
    row = _range_row(poles)
    return SimpleNamespace(
        pole_indices=row.pole_indices,
        bounds=np.repeat(row.bounds[:, None, :], 4, axis=1),
        phase_real=row.phase_real)


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

    union = _batch_rows(_union_row((2,)), (0, 1, 2, 3))
    assert union[1].shape == (4, 4, 6)
    assert np.isposinf(union[1][1:, :, 0]).all()


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

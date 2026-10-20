"""Current native on-shell request geometry only; no W/FFT/device work."""
import ast
from pathlib import Path

import numpy as np
import pytest

from common.units import RYD_TO_EV
from gw.ppm_sigma import on_shell_request
from tdagw.snapshot_runtime import _native_window_support_reads


@pytest.fixture(autouse=True)
def fp64_request_metadata():
    import jax
    with jax.enable_x64(True):
        yield


def test_current_shells_and_actual_native_auxiliary_reads_are_absolute_eV():
    e = np.array([[-.7, -.1, .3, .9], [-.6, -.2, .4, .8]])
    mu = -.25
    plan = dict(omega_range_ev=np.array([-2., 2.]), out_of_grid="cover")
    got = _native_window_support_reads(e, mu, plan)
    external = (e-mu)*RYD_TO_EV
    request = on_shell_request(e_eval_rel_ev=external, e_dft_rel_ev=external,
        omega_range_ev=plan["omega_range_ev"], out_of_grid="cover",
        irr_idx=np.arange(2), n_parent=2)
    expected = np.unique(np.r_[e.ravel()*RYD_TO_EV, (request.frequencies_ry+mu)*RYD_TO_EV])
    np.testing.assert_array_equal(got, expected)
    assert np.min(abs(got-mu*RYD_TO_EV)) < 1.e-14  # actual eval0/dft0 fallback
    assert got.min() >= e.min()*RYD_TO_EV-1.e-14
    assert got.max() <= e.max()*RYD_TO_EV+1.e-14


def test_changed_instantaneous_shells_change_reads_without_fullvirtual_table():
    e = np.array([[-.3, .2], [-.25, .3]])
    plan = dict(omega_range_ev=np.array([-100., 100.]), out_of_grid="cover")
    first = _native_window_support_reads(e, 0., plan)
    current = _native_window_support_reads(e+.02, 0., plan)
    np.testing.assert_allclose(current, first+.02*RYD_TO_EV, rtol=2.e-15, atol=2.e-15)
    assert current.size == e.size


@pytest.mark.parametrize("energy,mu", [(np.array([]), 0.), (np.array([1., 2.]), 0.),
    (np.array([[np.nan]]), 0.), (np.array([[.2]]), np.inf)])
def test_bad_current_request_refuses_before_expensive_builder(energy, mu):
    with pytest.raises(ValueError, match="td_native_support_reads"):
        _native_window_support_reads(energy, mu, dict(omega_range_ev=[-2., 2.], out_of_grid="cover"))


def test_evaluator_supplies_current_request_not_none_or_old_SC_recipe():
    import tdagw.snapshot_runtime as owner
    tree = ast.parse(Path(owner.__file__).read_text())
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "make_native_snapshot_evaluator")
    evaluator = next(n for n in factory.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
    calls = [n for n in ast.walk(evaluator) if isinstance(n, ast.Call)]
    request = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "_native_window_support_reads")
    builder = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "screen_general_shared_poles")
    assert request.lineno < builder.lineno
    snapshot = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "dict"
                    and any(k.arg == "support_reads_ev" for k in n.keywords))
    value = next(k.value for k in snapshot.keywords if k.arg == "support_reads_ev")
    assert isinstance(value, ast.Name) and value.id == "support_reads"
    assert "recipe_inputs" not in ast.get_source_segment(Path(owner.__file__).read_text(), evaluator)

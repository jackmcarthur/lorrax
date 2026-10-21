"""Exact instantaneous shells must not inherit a narrow requested SC grid."""
import ast
from pathlib import Path

import numpy as np
import pytest

from common.units import RYD_TO_EV
from gw import ppm_sigma as owner
from tdagw.snapshot_runtime import _native_window_support_reads


@pytest.fixture(autouse=True)
def fp64_metadata():
    import jax
    with jax.enable_x64(True):
        yield


def test_full44_request_does_not_become_static_at_legacy_quarter_eV_grid():
    # Accepted-reference-like shells: 30 occupied and 14 empty, all outside
    # the INITIAL requested SC range. SC itself grows that range privately.
    ef = -.3621754540614861
    row = np.r_[np.linspace(-.82, -.57, 30), np.linspace(-.342, -.18, 14)]
    energy = np.broadcast_to(row, (64, 44)).copy()
    energy += np.linspace(0., .0001, 64)[:, None]
    external = (energy-ef)*RYD_TO_EV
    old = owner.on_shell_request(e_eval_rel_ev=external, e_dft_rel_ev=external,
        omega_range_ev=np.array([-.25, 0., .25]), out_of_grid="cover",
        irr_idx=np.arange(64), n_parent=64)
    np.testing.assert_array_equal(old.frequencies_ry, [0.])
    current, receipt = owner.ordered_window_request(energy, ef)
    assert current.n_table == 1 and current.covered_kn.all()
    np.testing.assert_array_equal(current.tables_ry[0], external/RYD_TO_EV)
    np.testing.assert_array_equal(current.frequencies_ry[current.parent_index], current.parent_ry)
    assert current.aliases == {"pair": 0, "eval0": 0, "dft0": 0}
    np.testing.assert_allclose(receipt["domain_ev"], [external.min()-1., external.max()+1.])
    assert receipt["margin_ev"] == 1. and receipt["static_fallback_reads"] == 0
    assert not receipt["legacy_SC_grid_used"]
    assert not receipt["domain_boundaries_are_evaluation_frequencies"]
    reads = _native_window_support_reads(energy, ef,
        dict(omega_range_ev=[-.25, .25], out_of_grid="cover"))
    expected = np.unique(np.r_[energy.ravel()*RYD_TO_EV,
        (current.frequencies_ry+ef)*RYD_TO_EV])
    np.testing.assert_array_equal(reads, expected)


def test_same_helper_is_used_by_actual_sigma_and_support_reads():
    import tdagw.snapshot_runtime as runtime
    for module, function in [(owner, "_ordered_window_lines"),
                             (runtime, "_native_window_support_reads")]:
        tree = ast.parse(Path(module.__file__).read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        calls = [n.func.id for n in ast.walk(node) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name)]
        assert calls.count("ordered_window_request") == 1
        assert "on_shell_request" not in calls


def test_exact_shell_domain_can_include_origin_without_sampling_it():
    request, receipt = owner.ordered_window_request(np.array([[2., 3.]]), 0.)
    assert receipt["domain_ev"][0] == 0.
    assert not np.any(request.frequencies_ry == 0.)
    np.testing.assert_allclose(request.frequencies_ry, [2., 3.])


def test_changed_pair_or_static_fallback_is_refused(monkeypatch):
    from dataclasses import replace
    original = owner.on_shell_request
    def corrupt(**kwargs):
        request = original(**kwargs)
        return replace(request, tables_ry=np.zeros_like(request.tables_ry))
    monkeypatch.setattr(owner, "on_shell_request", corrupt)
    with pytest.raises(ValueError, match="current shells changed"):
        owner.ordered_window_request(np.array([[-.3, .4]]), 0.)


@pytest.mark.parametrize("energy,ef", [(np.array([]), 0.), (np.array([[np.nan]]), 0.),
                                        (np.array([[.2]]), np.inf)])
def test_invalid_current_shells_refuse(energy, ef):
    with pytest.raises(ValueError, match="ordered_window_request"):
        owner.ordered_window_request(energy, ef)

"""QP energy ownership across native and densified exciton input grids."""
import ast
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest


def _qp_owner(monkeypatch, coarse, corrected):
    # Exercise the production routing owner without starting MPI/JAX; only
    # immutable file reads and the pre-existing coarse reslicer are replaced.
    source = Path(__file__).parents[1] / "src/bse/bse_window.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "apply_eqp_to_bse_window")
    module = ModuleType("file_io.restart_bundle")
    module.read_metadata = lambda _path: {"energies": coarse}
    package = ModuleType("file_io")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "file_io", package)
    monkeypatch.setitem(sys.modules, "file_io.restart_bundle", module)
    calls = []

    def corrections(energies, *_args, **kwargs):
        np.testing.assert_array_equal(energies, coarse)
        assert kwargs["state_artifact_path"] == "restart.h5"
        return corrected.copy()

    def reslice(*_args, **_kwargs):
        calls.append("coarse reslice")
        v = np.column_stack((corrected[:, 1], np.full(len(coarse), -1e6)))
        c = np.column_stack((corrected[:, 2], np.full(len(coarse), +1e6)))
        return v, c, 2

    namespace = dict(DEFAULT_MODE="error", DEGENERACY_TOL_RY=1e-4,
                     _log0=lambda _: None, os=os, RYD_TO_EV=13.6056980659,
                     refuse_eqp_on_a_qp_wfn=lambda *_: None,
                     resolve_n_occ=lambda *_args, **_kwargs: 2,
                     apply_eqp_corrections=corrections,
                     apply_eqp_and_reslice_bands=reslice)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[node.name], calls


def test_densified_qp_payload_keeps_fine_grid_and_signed_padding(monkeypatch):
    coarse = np.array([[-3., -1., 2., 4.], [-2.8, -.9, 2.1, 4.2]])
    corrected = coarse + np.array([-.2, -.1, .4, .5])
    owner, calls = _qp_owner(monkeypatch, coarse, corrected)
    k = np.arange(7) / 7
    fine_v = np.column_stack((-1.1 + .2*np.cos(2*np.pi*k), np.full(7, -1e6)))
    fine_c = np.column_stack((2.4 + .3*np.sin(2*np.pi*k), np.full(7, +1e6)))
    data = dict(densified=True, n_val=1, n_cond=1, eps_v=fine_v,
                eps_c=fine_c, psi_c_X=np.zeros((7, 2, 1, 3)))
    before_v, before_c = fine_v.copy(), fine_c.copy()
    full = owner(data, "restart.h5", "eqp.dat", "gw.in", mesh_shape=(2, 2))
    np.testing.assert_array_equal(full, corrected)
    assert data["eps_v"] is fine_v and data["eps_c"] is fine_c
    np.testing.assert_array_equal(fine_v, before_v)
    np.testing.assert_array_equal(fine_c, before_c)
    assert not calls


def test_native_qp_reslice_and_unmarked_grid_mismatch_remain_guarded(monkeypatch):
    coarse = np.array([[-3., -1., 2., 4.], [-2.8, -.9, 2.1, 4.2]])
    corrected = coarse + np.array([-.2, -.1, .4, .5])
    owner, calls = _qp_owner(monkeypatch, coarse, corrected)
    data = dict(densified=False, n_val=1, n_cond=1, psi_c_X=np.zeros((2, 2, 1, 3)))
    full = owner(data, "restart.h5", "eqp.dat", "gw.in", mesh_shape=(2, 2))
    np.testing.assert_array_equal(full, corrected)
    np.testing.assert_array_equal(data["eps_v"][:, 0], corrected[:, 1])
    np.testing.assert_array_equal(data["eps_c"][:, 0], corrected[:, 2])
    assert np.all(data["eps_v"][:, 1] == -1e6)
    assert np.all(data["eps_c"][:, 1] == +1e6)
    assert calls == ["coarse reslice"]
    data["psi_c_X"] = np.zeros((7, 2, 1, 3))
    with pytest.raises(ValueError, match="native restart grid"):
        owner(data, "restart.h5", "eqp.dat", "gw.in", mesh_shape=(2, 2))


def test_continuum_matches_literal_same_k_transition_minimum():
    source = Path(__file__).parents[1] / "src/bse/exciton_bands.py"
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_continuum_threshold_ry")
    node.decorator_list = []  # Pure array identity; no JAX/device bootstrap.
    namespace = {"jnp": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    rng = np.random.default_rng(417)
    ev = rng.normal(-2., .5, (7, 2))
    ec = rng.normal(+2., .5, (3, 7, 3))
    # Deliberately attractive pad values prove only logical bands are read.
    ev_pad = np.pad(ev, ((0, 0), (0, 2)), constant_values=+1e6)
    ec_pad = np.pad(ec, ((0, 0), (0, 0), (0, 1)), constant_values=-1e6)
    actual = namespace[node.name](ec_pad, ev_pad, n_val=2, n_cond=3)
    expected = np.array([min(ec[q, k, c]-ev[k, v]
                            for k in range(7) for c in range(3) for v in range(2))
                         for q in range(3)])
    np.testing.assert_array_equal(actual, expected)

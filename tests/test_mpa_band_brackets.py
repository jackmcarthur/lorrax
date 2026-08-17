"""Estimator-neutral gates for MPA's cumulative band-bracket machinery."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys
import types

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from gw.band_extrapolation import BandBracketPlan, trivial_plan
from gw.minimax_screening import MinimaxNodes
from gw.ppm_windows import _SigmaWindow


class _Slices:
    b0 = 0
    b2 = 2
    b4 = 6
    b4_sigma = 6
    sigma_sum = slice(0, 6)
    sigma = slice(0, 2)
    nb_sigma_sum = 6


class _Wfns:
    slices = _Slices()

    def __init__(self):
        self.enk = np.arange(6, dtype=np.float64)[None]
        self.occ = np.asarray([[1, 1, 0, 0, 0, 0]], dtype=np.float64)
        self._xn = jnp.zeros((1, 1, 1, 6), dtype=jnp.complex128)
        self._yr = jnp.zeros((1, 6, 1, 1), dtype=jnp.complex128)
        self._xr = jnp.ones((1, 2, 1, 1), dtype=jnp.complex128)
        self._yn = jnp.ones((1, 1, 1, 2), dtype=jnp.complex128)

    def xn(self, _):
        return self._xn

    def yr(self, _):
        return self._yr

    def xr(self, _):
        return self._xr

    def yn(self, _):
        return self._yn


def _window():
    win = _SigmaWindow(
        name="unit", nodes=MinimaxNodes(
            t=jnp.asarray([0.25 + 0.0j]),
            alpha=jnp.asarray([0.75 + 0.0j])),
        mask_A=np.ones((1, 6), bool), E_ref_A=0.0, E_ref_B=0.0,
        omega_sign=1, project="full", prefactor=1.0)
    return types.SimpleNamespace(
        window=win, E_A=jnp.arange(6, dtype=jnp.float64)[None],
        omega_abs=np.asarray([0.1, 0.4]), omega_idx=np.asarray([0, 1]),
        pole_indices=np.asarray([0]),
        bounds=np.asarray([[0.0, np.inf, -np.inf, -np.inf,
                            np.inf, np.inf]]),
        phase_real=np.asarray([False]))


def _load_mpa_sigma_without_io_stack():
    """Load the executor leaf without importing the optional HDF5 stack."""
    name = "_test_mpa_band_bracket_executor"
    if name in sys.modules:
        return sys.modules[name]
    store = types.ModuleType("file_io.mpa_store")
    store.PoleReader = type("PoleReader", (), {})
    store.open_pole_reader = lambda *a, **k: None
    store.validate_fit_store = lambda *a, **k: {"n_p": 1}
    file_io = types.ModuleType("file_io")
    file_io.__path__ = []
    mpa_pkg = types.ModuleType("gw.mpa")
    mpa_pkg.__path__ = []
    windows = types.ModuleType("gw.mpa.sigma_windows")
    windows.build_shared_sigma_windows = lambda *a, **k: None
    windows.summarize_sigma_poles = lambda *a, **k: None
    names = ("file_io", "file_io.mpa_store", "gw.mpa",
             "gw.mpa.sigma_windows")
    old = {key: sys.modules.get(key) for key in names}
    sys.modules["file_io"] = file_io
    sys.modules["file_io.mpa_store"] = store
    sys.modules["gw.mpa"] = mpa_pkg
    sys.modules["gw.mpa.sigma_windows"] = windows
    try:
        path = Path(__file__).parents[1] / "src" / "gw" / "mpa" / "sigma.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "gw.mpa"
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for key, value in old.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def _run(monkeypatch, band_plan):
    mpa_sigma = _load_mpa_sigma_without_io_stack()
    calls = []

    def factory(*, mesh_xy, kgrid, brackets=None):
        del mesh_xy, kgrid
        calls.append(("factory", brackets))

        def tau(*args):
            del args
            calls.append(("tau", brackets))
            if brackets is None:
                return jnp.full((1, 2, 2), sum(range(1, 7)),
                                dtype=jnp.complex128)
            return jnp.stack([
                jnp.full((1, 2, 2), sum(range(lo + 1, hi + 1)),
                         dtype=jnp.complex128)
                for lo, hi in brackets
            ])

        return tau

    monkeypatch.setattr(mpa_sigma, "get_shared_sigma_tau_kernel", factory)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    batches = [(0, jnp.ones((1, 1, 1, 1), dtype=jnp.complex128),
                jnp.ones((1, 1, 1, 1), dtype=jnp.complex128))]
    out = mpa_sigma._integrate_sigma_batches(
        _Wfns(), batches, 1, [_window()], np.asarray([0.1, 0.4]),
        types.SimpleNamespace(nkx=1, nky=1, nkz=1), mesh,
        pole_batch_size=1, band_plan=band_plan, print_fn=lambda _: None)
    return out, calls


def _plan(bounds):
    counts = tuple(hi for _, hi in bounds)
    return BandBracketPlan(
        bounds=tuple(bounds), counts=counts, requested=counts,
        n_occ=2, n_cond=4, mean_energy_ev=(0.0,) * len(bounds),
        enabled=len(bounds) > 1)


def test_one_bracket_is_bit_identical_to_unbracketed_mpa(monkeypatch):
    baseline, base_calls = _run(monkeypatch, None)
    one, one_calls = _run(monkeypatch, trivial_plan(6, 2, 6))
    assert np.array_equal(np.asarray(baseline.sigma_c_kij),
                          np.asarray(one.sigma_c_kij[0]))
    assert sum(c[0] == "tau" for c in base_calls) == 1
    assert sum(c[0] == "tau" for c in one_calls) == 1
    assert one.band_counts == (6,)


def test_three_cumulative_points_partition_every_z_sample(monkeypatch):
    plan = _plan(((0, 3), (3, 5), (5, 6)))
    out, calls = _run(monkeypatch, plan)
    baseline, baseline_calls = _run(monkeypatch, None)
    cube = np.asarray(out.sigma_c_kij)
    assert cube.shape == (3, 2, 1, 2, 2)
    assert out.band_counts == plan.counts
    assert np.isfinite(cube).all()
    # Both z/omega samples independently recover the single full-band sum.
    np.testing.assert_allclose(
        cube[-1], np.asarray(baseline.sigma_c_kij),
        rtol=2e-15, atol=2e-15)
    assert sum(c[0] == "tau" for c in calls) == 1
    assert sum(c[0] == "tau" for c in baseline_calls) == 1
    assert sum(hi - lo for lo, hi in plan.bounds) == _Slices.nb_sigma_sum


def test_bracket_count_and_widths_are_runtime_data(monkeypatch):
    plan = _plan(((0, 1), (1, 2), (2, 4), (4, 6)))
    out, calls = _run(monkeypatch, plan)
    assert np.asarray(out.sigma_c_kij).shape[0] == 4
    assert out.band_counts == (1, 2, 4, 6)
    assert sum(c[0] == "tau" for c in calls) == 1


def test_mpa_dispatch_selects_full_point_and_names_no_estimator():
    import gw.sigma_dispatch as dispatch

    src = inspect.getsource(dispatch.compute_sigma_xc)
    mpa = src[src.index("if mode is ComputeMode.MPA:"):]
    mpa = mpa[:mpa.index("THE EXHAUSTIVENESS SEAM")]
    assert "_band_count_point(" in mpa
    assert "body.sigma_c_kij.shape[0] - 1" in mpa
    for forbidden in (
            "fit_band_extrapolation(", "extrapolation_weights(",
            "_report_band_extrapolation(", "_extrapolated_point("):
        assert forbidden not in mpa, (
            f"MPA machinery must not select an estimator: found {forbidden}")


def test_runtime_occupation_half_uses_the_named_refusal():
    from gw.band_extrapolation import (
        MPA_BRACKET_INSULATOR_GATE,
        refuse_mpa_bracket_metallic_occupations)

    said = []
    worst = refuse_mpa_bracket_metallic_occupations(
        np.asarray([[[1.0, 1.0, 0.0, 0.0]]]),
        band_lo=0, band_hi=4, source="integer-WFN", print_fn=said.append)
    assert worst == 0.0
    assert MPA_BRACKET_INSULATOR_GATE in said[0]

    with pytest.raises(NotImplementedError) as exc:
        refuse_mpa_bracket_metallic_occupations(
            np.asarray([[[1.0, 0.75, 0.25, 0.0]]]),
            band_lo=0, band_hi=4, source="metal-WFN")
    message = str(exc.value)
    assert MPA_BRACKET_INSULATOR_GATE in message
    for field in ("got:", "want:", "fix:", "why:", "doc:"):
        assert field in message
    assert "feat/occupation-support-guard-2026-08-16" in message


def test_declared_metal_half_uses_the_same_named_refusal():
    from gw.gw_config import ComputeMode
    from gw.sigma_dispatch import compute_sigma_xc

    cfg = types.SimpleNamespace(
        sigma=types.SimpleNamespace(band_extrapolation=True),
        mpa=types.SimpleNamespace(material_class="metal"))
    with pytest.raises(NotImplementedError) as exc:
        compute_sigma_xc(
            ComputeMode.MPA, wfns=None, V_q=None, W_by_role={},
            e_qp_ev=None, static_head_terms=None, head_resolver=None,
            quad=None, config=cfg, meta=None, mesh_xy=None, sym=None,
            wfn=None, band_slices=None, input_dir=".")
    message = str(exc.value)
    from gw.band_extrapolation import MPA_BRACKET_INSULATOR_GATE
    assert MPA_BRACKET_INSULATOR_GATE in message
    for field in ("got:", "want:", "fix:", "why:", "doc:"):
        assert field in message


def test_low_level_bracket_call_refuses_an_occupation_state():
    mpa_sigma = _load_mpa_sigma_without_io_stack()
    with pytest.raises(NotImplementedError) as exc:
        mpa_sigma.compute_sigma_c_mpa_omega_grid(
            _Wfns(), "unused-store", types.SimpleNamespace(
                b_id_4_sigma_user=6), object(),
            omega_grid_ry=np.asarray([0.1]), efermi_ry=0.0,
            regularization_width_ry=0.01, target_error=1e-6,
            max_rank=4, crossing_max_nodes=8,
            occupation_state=types.SimpleNamespace(),
            band_plan=trivial_plan(6, 2, 6))
    message = str(exc.value)
    assert mpa_sigma.MPA_BRACKET_INSULATOR_GATE in message
    assert "occupation_state" in message

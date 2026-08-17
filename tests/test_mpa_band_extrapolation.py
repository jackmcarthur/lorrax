"""MPA consumer gates for the shared band-bracket/OLS path."""

from __future__ import annotations

import types
import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np

from gw.band_extrapolation import (
    BandBracketPlan,
    extrapolation_weights,
    fit_band_extrapolation,
    trivial_plan,
)
from gw.efermi import OCCUPATION_WINDOW_THRESHOLD_DEFAULT
from gw.minimax_screening import MinimaxNodes
from gw.ppm_windows import _SigmaWindow


class _Slices:
    b0 = 0
    b2 = 2
    b4 = 6
    sigma_sum = slice(0, 6)
    sigma = slice(0, 2)
    nb_sigma_sum = 6


class _Wfns:
    slices = _Slices()

    def __init__(self):
        self.enk = np.arange(6, dtype=np.float64)[None]
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
        omega_abs=np.asarray([0.1]), omega_idx=np.asarray([0]),
        pole_indices=np.asarray([0]),
        bounds=np.asarray([[0.0, np.inf, -np.inf, -np.inf,
                            np.inf, np.inf]]),
        phase_real=np.asarray([False]))


def _load_mpa_sigma_without_io_stack():
    """Load the executor leaf without importing file_io's h5py service.

    The J070 pytest image deliberately has no h5py.  This unit exercises the
    in-memory executor only, so stub the three store names its module imports;
    production and the acceptance leg import the real store normally.
    """
    name = "_test_mpa_sigma_executor"
    if name in sys.modules:
        return sys.modules[name]
    store = types.ModuleType("file_io.mpa_store")
    store.PoleReader = type("PoleReader", (), {})
    store.open_pole_reader = lambda *a, **k: None
    store.validate_fit_store = lambda *a, **k: None
    file_io = types.ModuleType("file_io")
    file_io.__path__ = []
    mpa_pkg = types.ModuleType("gw.mpa")
    mpa_pkg.__path__ = []
    windows = types.ModuleType("gw.mpa.sigma_windows")
    windows.OCCUPATION_WINDOW_THRESHOLD_DEFAULT = (
        OCCUPATION_WINDOW_THRESHOLD_DEFAULT)
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
            # A deterministic linear band contribution.  Static slicing in
            # the real kernel produces these same disjoint increment sums.
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
        _Wfns(), batches, 1, [_window()], np.asarray([0.1]),
        types.SimpleNamespace(nkx=1, nky=1, nkz=1), mesh,
        pole_batch_size=1, band_plan=band_plan, print_fn=lambda _: None)
    return out, calls


def test_mpa_one_bracket_is_bit_identical_to_the_unbracketed_baseline(
        monkeypatch):
    baseline, base_calls = _run(monkeypatch, None)
    one, one_calls = _run(monkeypatch, trivial_plan(6, 2, 6))
    assert np.array_equal(np.asarray(baseline.sigma_c_kij),
                          np.asarray(one.sigma_c_kij[0]))
    assert sum(c[0] == "tau" for c in base_calls) == 1
    assert sum(c[0] == "tau" for c in one_calls) == 1


def test_mpa_three_increments_are_cumulative_finite_and_dispatch_neutral(
        monkeypatch):
    plan = BandBracketPlan(
        bounds=((0, 3), (3, 5), (5, 6)), counts=(3, 5, 6),
        requested=(3, 5, 6), n_occ=2, n_cond=4,
        mean_energy_ev=(0.0, 0.0, 0.0), enabled=True)
    out, calls = _run(monkeypatch, plan)
    baseline, baseline_calls = _run(monkeypatch, None)
    cube = np.asarray(out.sigma_c_kij)
    assert cube.shape == (3, 1, 1, 2, 2)
    assert np.isfinite(cube).all()
    np.testing.assert_allclose(cube[-1], np.asarray(baseline.sigma_c_kij),
                               rtol=2e-15, atol=2e-15)
    # One shared tau dispatch, not one dispatch per bracket.  The widths are
    # a true partition, so the band contraction visits six bands total.
    assert sum(c[0] == "tau" for c in calls) == 1
    assert sum(c[0] == "tau" for c in baseline_calls) == 1
    assert sum(hi - lo for lo, hi in plan.bounds) == _Slices.nb_sigma_sum

    fit = fit_band_extrapolation(plan.counts, cube)
    assert np.isfinite(fit.s_inf).all()
    combined = np.tensordot(extrapolation_weights(out.band_counts), cube,
                            axes=(0, 0))
    assert np.isfinite(combined).all()


def test_mpa_reuses_the_ppm_ols_and_trust_consumer():
    import inspect
    import gw.sigma_dispatch as dispatch

    src = inspect.getsource(dispatch.compute_sigma_xc)
    mpa = src[src.index("if mode is ComputeMode.MPA:"):]
    mpa = mpa[:mpa.index("THE EXHAUSTIVENESS SEAM")]
    assert "_extrapolated_point(" in mpa
    assert "_report_band_extrapolation(" in mpa
    assert "fit_band_extrapolation(" not in mpa, (
        "MPA must reuse the shared PPM OLS/trust report, not copy its math")

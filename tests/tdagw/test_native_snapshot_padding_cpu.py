"""Logical band support versus native inert padding; CPU owner-assembly checks."""
from dataclasses import replace
from types import SimpleNamespace as NS

import runtime  # noqa: F401
import jax
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from tdagw import snapshot_runtime as sr

jax.config.update("jax_enable_x64", True)


def _source(monkeypatch, logical):
    from tests.test_tdagw_full_microscopic_cpu import _native_port_fixture
    case = _native_port_fixture(monkeypatch)
    fixture, nk = case["fixture"], case["nk"]
    padded = (logical+3)//4*4
    monkeypatch.setattr(fixture, "NBF", logical)
    monkeypatch.setattr(fixture, "NB_PAD", padded)
    energy = np.broadcast_to(np.linspace(.6, 4., logical), (nk, logical)).copy()
    energy[:, :3] = [-1., -.2, .3]
    occ = np.zeros_like(energy)
    occ[:, :2] = 1.
    rng = np.random.default_rng(675)
    psi = .1*(rng.normal(size=(nk, logical, 2, 4))+1j*rng.normal(size=(nk, logical, 2, 4)))
    wfns, meta = fixture._gw_bundle(case["mesh"], psi, energy, occ,
                                    case["wfns"].green_parent.plan)
    meta.mu_basis = case["meta"].mu_basis
    meta.shared_pole_recipe = {}
    meta.shared_pole_census = dict(mu_ry=0., state_capacity=1., k_weights=np.full(nk, 1/nk))
    V = jax.device_put(np.broadcast_to(np.eye(4), (nk, 4, 4)).astype(complex),
                      NamedSharding(case["mesh"], P(None, "x", "y")))
    config = fixture._config(np.array([-3., 4.]))
    config.sigma.band_extrapolation = False
    source = dict(wfns=wfns, meta=meta, V_q=V, config=config,
        screening_inputs=dict(wfns=wfns, V_q=V, meta=meta, config=config,
                              kwargs=dict(sym=case["sym"])))
    return source, case["mesh"], config


@pytest.mark.parametrize("logical", [3, 750])
def test_padded_snapshot_reaches_actual_response_and_sigma_owner_assembly(monkeypatch, tmp_path, logical):
    from ffi import fft
    import gw.shared_pole_screening as screening
    import gw.shared_pole_recipe as recipe
    import gw.ppm_sigma as sigma

    source, mesh, config = _source(monkeypatch, logical)
    prepared = sr.prepare_native_source(source, mesh)
    padded = source["wfns"].enk.shape[1]
    assert prepared["nb"] == logical and prepared["nb_padded"] == padded
    assert prepared["energy"].shape[1] == logical
    np.testing.assert_array_equal(prepared["energy_padded"], source["wfns"].enk)
    captured = {}

    class ReachedCensus(Exception):
        pass

    def census(current, meta, *, occupation_state, **kwargs):
        captured["response_energy"] = np.asarray(current.enk)
        captured["response_rho_diagonal"] = occupation_state.f_kn.copy()
        raise ReachedCensus

    original_screen = screening.screen_general_shared_poles
    def screen(*args, **kwargs):
        captured["snapshot"] = kwargs["snapshot"]
        with pytest.raises(ReachedCensus):
            original_screen(*args, **kwargs)
        return dict(shared_pole={}, meta=source["meta"], receipt={})

    def ordered(source_handle, plan, *, enk, rho, pop, **kwargs):
        # Actual window_sigma_c has validated its full padded energy shape,
        # constructed current populations and rotated both native faces.
        captured.update(sigma_energy=enk, sigma_rho=rho, sigma_pop=pop)
        pair = np.zeros((2, prepared["nk"], 2, 2), complex)
        branches = dict(cond=pair, val=pair)
        return dict(active_branch_pair=branches, branch_pair=branches, native_receipt={})

    monkeypatch.setattr(recipe, "bind_shared_pole_census", census)
    monkeypatch.setattr(screening, "screen_general_shared_poles", screen)
    monkeypatch.setattr(sigma, "shared_pole_window_source", lambda *a, **kw:
        dict(member="shared_pole", representation="ordered_complex", model={},
             meta=source["meta"], efermi_ry=0.))
    monkeypatch.setattr(sigma, "_ordered_window_lines", ordered)
    rho = np.broadcast_to(np.array([[.8, .1j], [-.1j, .2]]), (prepared["nk"], 2, 2)).copy()
    e = prepared["energy"][:, 1:3]+np.array([-.03, .04])
    basis = dict(U=np.broadcast_to(np.eye(2), rho.shape), e=e, rho=rho)
    with fft.xla_reference():
        evaluate, _ = sr.make_native_snapshot_evaluator(source, prepared,
            dict(b_lo=1, n_active=2, spm_scratch=str(tmp_path)))
        evaluate(basis, equilibrium=True)
    for key in ("response_energy", "sigma_energy"):
        assert captured[key].shape == (prepared["nk"], padded)
        np.testing.assert_array_equal(captured[key][:, logical:], prepared["energy_padded"][:, logical:])
        np.testing.assert_array_equal(captured[key][:, 1:3], e)
    np.testing.assert_array_equal(captured["sigma_rho"], rho)
    assert not np.any(captured["response_rho_diagonal"][:, logical:])
    assert not np.any(captured["sigma_pop"][:, logical:])
    assert captured["snapshot"]["response_active_slice"] == slice(1, 3)


@pytest.mark.parametrize("bad", ["occupation", "orbital"])
def test_native_padding_refuses_nonzero_support(monkeypatch, bad):
    source, mesh, _ = _source(monkeypatch, 3)
    wfns = source["wfns"]
    if bad == "occupation":
        source["wfns"] = replace(wfns, occ=wfns.occ.at[:, 3].set(.1))
    else:
        carrier = replace(wfns.green_parent,
            psi_nmu=wfns.green_parent.psi_nmu.at[:, 3].set(.1))
        source["wfns"] = replace(wfns, green_parent=carrier)
    with pytest.raises(ValueError, match="padded occupations/orbitals must be exact zero"):
        sr.prepare_native_source(source, mesh)


def test_loaded_physical_bands_beyond_consumer_caps_are_not_padding(monkeypatch):
    source, mesh, _ = _source(monkeypatch, 8)
    wfns = source["wfns"]
    source["wfns"] = replace(wfns, slices=replace(wfns.slices,
        b4_chi=6, b4_sigma=6))
    assert np.any(np.asarray(wfns.green_parent.psi_nmu)[:, 6:8] != 0)
    prepared = sr.prepare_native_source(source, mesh)
    assert prepared["nb"] == prepared["nb_padded"] == 8
    assert prepared["nb_chi"] == prepared["nb_sigma"] == 6
    np.testing.assert_array_equal(prepared["energy"], wfns.enk)
    np.testing.assert_array_equal(prepared["occupations"], wfns.occ)

"""GN/HL one-pole conversion and shared-MPA routing seam."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw.band_extrapolation import trivial_plan
from gw.ppm_sigma import PPMBuildResult, SigmaOmegaResult
from gw.wavefunction_bundle import BandSlices


def _mesh():
    return Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1),
                ("x", "y"))


def test_ppm_fit_is_persisted_then_consumed_by_the_mpa_route(monkeypatch):
    from file_io import mpa_store
    from gw import ppm_sigma
    from gw.mpa import sigma as mpa_sigma

    mesh = _mesh()
    slices = BandSlices.from_band_edges(
        0, 0, 1, 2, 2, b4_chi=2, b4_sigma=2)
    wfns = SimpleNamespace(
        slices=slices,
        enk=jnp.asarray([[-0.4, 0.6]], dtype=jnp.float64),
        occ=jnp.asarray([[1.0, 0.0]], dtype=jnp.float64))
    B = jnp.asarray([[[0.7 + 0.1j, -0.2], [0.3j, 0.5]]])
    D = jnp.asarray([[[0.05j, 0.7], [-0.02j, -0.1]]])
    Omega = jnp.asarray([[[0.8, 1.0], [1.2, 1.4]]])
    ppm = PPMBuildResult(
        omega_p=0.5,
        Wc0_q=-2.0 * B / Omega,
        B_q=B,
        Omega_q=Omega,
        valid_mask_q=jnp.asarray([[[True, False], [True, True]]]),
        unfulfilled_fraction=0.25,
        n_nodes_static=8,
        B_odd_q=D)
    meta = SimpleNamespace(
        nk_tot=1, nkx=1, nky=1, nkz=1, n_rmu=2,
        b_id_4_sigma_user=2)
    ppm_cfg = SimpleNamespace(invalid_mode="zero")
    sigma_cfg = SimpleNamespace(
        regularization_ev=0.25,
        regularization_floor_ev=0.0,
        window_edge_factor=1.5,
        fermi_reference="midgap",
        quadrature_eps=3.0e-5,
        quadrature_reduction_seconds=42.0,
        omega_step_ev=0.5)
    mpa_cfg = SimpleNamespace(sigma_max_nodes=91, pole_batch_size=4)
    plan = trivial_plan(2, 1, 2)
    omega_grid = np.asarray([-0.1, 0.2])
    captured = {}

    def fake_write(path, Omega_p, B_p, **kwargs):
        captured["store"] = (
            path, np.asarray(Omega_p), np.asarray(B_p), kwargs)
        return {"complete": True}

    def fake_mpa(wfns_arg, path, meta_arg, mesh_arg, **kwargs):
        captured["mpa"] = (wfns_arg, path, meta_arg, mesh_arg, kwargs)
        shape = (1, omega_grid.size, 1, 2, 2)
        values = jax.device_put(
            np.ones(shape, dtype=np.complex128),
            NamedSharding(mesh, P(None, None, None, "x", "y")))
        return SigmaOmegaResult(
            omega_ry=omega_grid,
            omega_ev=omega_grid * 13.605693122994,
            sigma_c_kij=values,
            sigma_c_odd_kij=0.125 * values,
            band_counts=plan.counts)

    monkeypatch.setattr(
        mpa_store, "write_complete_pole_store_collective", fake_write)
    monkeypatch.setattr(
        mpa_sigma, "compute_sigma_c_mpa_omega_grid", fake_mpa)

    result = ppm_sigma.compute_sigma_c_ppm_omega_grid(
        wfns, ppm, meta, mesh,
        ppm_cfg=ppm_cfg,
        sigma_cfg=sigma_cfg,
        mpa_cfg=mpa_cfg,
        omega_grid_ry=omega_grid,
        ansatz="gn_ppm",
        fit_store_path="/tmp/one-pole.h5",
        screening_diagrams="w_rpa",
        quadrature_cache_dir="/tmp/rule-cache",
        plan=plan,
        print_fn=lambda *_args, **_kwargs: None)

    path, stored_Omega, stored_B, store_kw = captured["store"]
    assert path == "/tmp/one-pole.h5"
    assert stored_Omega.shape == stored_B.shape == (1, 1, 2, 2)
    assert stored_Omega[0, 0, 0, 1] == 0.0
    assert stored_B[0, 0, 0, 1] == 0.0
    stored_D = np.asarray(store_kw["B_odd_p"])
    assert stored_D.shape == stored_B.shape
    assert stored_D[0, 0, 0, 1] == 0.0
    assert stored_D[0, 0, 0, 0] == D[0, 0, 0]
    assert store_kw["provenance"]["fit_protocol"] == "two_point_ppm"
    assert store_kw["provenance"]["pole_model"] == "gn_ppm"
    assert store_kw["provenance"]["ppm_invalid_mode"] == "zero"

    _, mpa_path, _, _, mpa_kw = captured["mpa"]
    assert mpa_path == path
    assert mpa_kw["quadrature_eps"] == 3.0e-5
    assert mpa_kw["quadrature_reduction_seconds"] == 42.0
    assert mpa_kw["quadrature_cache_dir"] == "/tmp/rule-cache"
    assert mpa_kw["pair_ceiling"] == 91
    assert mpa_kw["pole_batch_size"] == 4
    assert mpa_kw["band_brackets"] == plan.bounds
    assert mpa_kw["band_counts"] == plan.counts
    assert len(mpa_kw["sigma_branches"]) == 4
    assert result.band_counts == plan.counts
    assert result.sigma_c_kij.shape == (1, 2, 1, 2, 2)
    np.testing.assert_array_equal(
        np.asarray(result.sigma_c_odd_kij),
        0.125 * np.asarray(result.sigma_c_kij))

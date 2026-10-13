"""Stable photon API, planner, and sample/restart glue on small CPU matrices.

Run on a compute node. Checked service fixtures exercise the real response
programs without a native GPU provider. The sample test executes the actual
nested production ``solve`` body with a bounded in-memory bank and raw chi;
no response integration, scientific file, fit, or pole constructor is used.
"""
import ast
import inspect
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from tests.test_compile_stability_cpu import _CheckedPlan, _matmul, _mesh
from tests.test_photon_response_stable_numpy import fixture, ordered_coefficients, stable

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def photon_api(monkeypatch):
    import distrib_la
    from gw import response_bank as rb

    monkeypatch.setattr(distrib_la, "plan", _CheckedPlan)
    monkeypatch.setattr(distrib_la, "matmul", _matmul)
    rb._response_programs.cache_clear()
    mesh = _mesh()
    meta = SimpleNamespace(nk_tot=4, nspin=1, nspinor_wfnfile=2, cell_volume=2.3)
    face = NamedSharding(mesh, P(None, "x", "y"))
    make = lambda n: rb.response_algebra(
        meta, {"linalg": "distributed"}, mesh_xy=mesh, n=n, ordered=True, photon=True)
    with mesh:
        yield SimpleNamespace(mesh=mesh, meta=meta, make=make, rb=rb,
                              put=lambda a: jax.device_put(np.asarray(a), face))
    rb._response_programs.cache_clear()


def test_public_pair_split_slope_and_ordered_moments(photon_api):
    api = photon_api
    dyson, slope, moments, algebra = api.make(3)
    v, c, p, d = fixture()
    v2, _, _, _ = fixture(False)
    vs = np.stack((v, v2))
    ps = np.stack((p, .7*p + .3*d))
    ds = np.stack((d, .8*d.T))
    pref = algebra["prefactor"]
    contact = api.put(c[None] / api.meta.cell_volume)
    vj, chi, dchi = (api.put(x) for x in (vs, ps/pref, ds/pref))
    r = dyson.infinity(vj, contact)
    expected_r = np.stack([np.linalg.solve(np.eye(3)+vi@c, vi) for vi in vs])
    np.testing.assert_allclose(r, expected_r, atol=5e-14, rtol=2e-12)
    expected = np.stack([stable(vi, pi, c, ri) for vi, pi, ri in zip(vs, ps, expected_r)])
    value, paired_slope = dyson.pair("face")(vj, chi, dchi, contact, r)
    np.testing.assert_allclose(value, expected, atol=5e-14, rtol=2e-12)
    np.testing.assert_allclose(dyson.value(vj, chi, contact, r), expected, atol=5e-14, rtol=2e-12)
    full = expected + expected_r
    expected_slope = full @ ds @ full
    np.testing.assert_allclose(paired_slope, expected_slope, atol=5e-14, rtol=2e-12)
    np.testing.assert_allclose(slope(value+r, dchi), expected_slope, atol=5e-14, rtol=2e-12)
    ks = tuple(np.stack((k, .8*k)) for k in (p, d, .6*p.T, .7*d.conj().T))
    constant, *stored = moments(vj, api.put(ks[1]), api.put(ks[3]),
                                api.put(ks[0]), api.put(ks[2]), contact)
    np.testing.assert_allclose(constant, expected_r-vs, atol=5e-14, rtol=2e-12)
    words = [ordered_coefficients(ri, [k[i] for k in ks]) for i, ri in enumerate(expected_r)]
    for j, actual in enumerate(stored):
        np.testing.assert_allclose(actual, .5*np.stack([row[j] for row in words]),
                                   atol=5e-14, rtol=2e-12)


def test_public_cancellation_keeps_actual_infinity(photon_api):
    api = photon_api
    dyson, slope, _, algebra = api.make(1)
    v = api.put([[[float(2**54)]]])
    c = api.put([[[1/api.meta.cell_volume]]])
    p = api.put([[[float(2**-8)/algebra["prefactor"]]]])
    dp = api.put([[[1/algebra["prefactor"]]]])
    r = dyson.infinity(v, c)
    value, derivative = dyson.pair("face")(v, p, dp, c, r)
    expected_full = 1/(1-2**-8)
    assert float(np.linalg.norm(np.asarray((r-v)+v-r))) > .5
    np.testing.assert_allclose(value, expected_full-1, atol=1e-14, rtol=2e-12)
    np.testing.assert_allclose(derivative, expected_full**2, atol=1e-13, rtol=2e-12)
    np.testing.assert_allclose(slope(value+r, dp), expected_full**2, atol=1e-13, rtol=2e-12)


def test_photon_planner_compiles_pair_and_split_arities(photon_api, monkeypatch):
    api = photon_api
    dyson, slope, _, _ = api.make(3)
    monkeypatch.setattr(api.rb, "response_dense_workspace", lambda *a, **k: {"total": 0})
    v = api.put(np.eye(3, dtype=complex)[None])
    contact = api.put(np.eye(3, dtype=complex)[None]*.1)
    for selected in (dyson, SimpleNamespace(value=dyson.value, pair=lambda _: None)):
        residence, workspace = api.rb._dyson_phase(
            selected, slope, v, v, api.mesh, "distributed", nq=1, n=3, extra=(contact,))
        assert residence > 0 and workspace >= 0


def _production_sample_solve(rb, namespace):
    """Execute the existing nested owner; copy no sample or restart equations."""
    tree = ast.parse(inspect.getsource(rb.produce_sample_bank))
    candidates = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "solve"]
    assert len(candidates) == 1, "produce_sample_bank must expose its one solve stage"
    module = ast.fix_missing_locations(ast.Module(body=candidates, type_ignores=[]))
    namespace = dict(namespace, __package__="gw")
    exec(compile(module, inspect.getsourcefile(rb.produce_sample_bank), "exec"), namespace)
    return namespace["solve"]


@pytest.mark.parametrize("phase", ("fresh", "slope_only_resume", "partner"))
@pytest.mark.parametrize("with_head", (False, True))
def test_sample_resume_and_direct_head_dynamic_static_separation(photon_api, phase, with_head):
    api = photon_api
    from gw.photon_layout import PhotonBasisLayout

    layout = PhotonBasisLayout.from_centroid_extents(1, 1, api.mesh, packed=True)
    assert layout.packed_extent == 4
    v3, c3, p3, d3 = fixture()
    v, c, p, d = (np.zeros((4, 4), complex) for _ in range(4))
    v[:3, :3], c[:3, :3], p[:3, :3], d[:3, :3] = v3, c3, p3, d3
    v[3, 3], c[3, 3], p[3, 3], d[3, 3] = .9, .07, .04, .03
    dyson, slope, _, algebra = api.make(4)
    r = np.linalg.solve(np.eye(4)+v@c, v)
    partner = phase == "partner"
    effective_p, effective_d = (p.conj(), d.conj()) if partner else (p, d)
    base = stable(v, effective_p, c, r)
    head_value = .04*np.eye(4, dtype=complex)
    head_value[0, 2] = .011+.009j
    head_slope = .023*np.eye(4, dtype=complex)
    static_head = .8*np.eye(4, dtype=complex)
    direct = None
    if with_head:
        direct = dict(constant=static_head, Wc=np.array([head_value]),
                      Wc_minus_q=np.array([1.2*head_value.conj()]),
                      dWc_ds=np.array([head_slope]),
                      dWc_minus_q_ds=np.array([.7*head_slope]),
                      gamma_vectors=(jax.device_put(np.eye(4, dtype=complex), NamedSharding(api.mesh, P(None, "x"))),
                                     jax.device_put(np.eye(4, dtype=complex), NamedSharding(api.mesh, P(None, "y")))))
    dynamic = (direct["Wc_minus_q" if partner else "Wc"][0] if with_head else np.zeros_like(v))
    dynamic_slope = (direct["dWc_minus_q_ds" if partner else "dWc_ds"][0]
                     if with_head else np.zeros_like(v))
    saved = api.put((base+dynamic)[None])
    reads, calls = [], []

    def read_bank(*args, **kwargs):
        reads.append(kwargs)
        return {"Wc": saved[:, None]}

    def execute(kernel, args, stage):
        calls.append((stage, args))
        return kernel(*args)

    vj = api.put(v[None])
    namespace = dict(np=np, jnp=jnp, time=time, mirror=None, partner_qids=np.array([0]),
        qids=np.array([0]), row_index={0: 0}, roots=vj, held=vj, vertex=object(),
        bank_io={"photon_winf": api.put(r[None]), "photon_layout": layout},
        read_constant=lambda *a: api.put((r-v+(static_head if with_head else 0))[None]),
        direct_head=direct, contact=api.put(c[None]/api.meta.cell_volume),
        mesh_xy=api.mesh, receipt={"seconds": {}}, dyson=dyson, solve_slope=slope,
        execute=execute, read_shared_pole_bank=read_bank, meta=api.meta,
        header={}, z=np.array([.7j]), _photon_sample_norms=lambda *a: None,
        _self_negative=lambda *a: False, tr_odd_census=False)
    solve = _production_sample_solve(api.rb, namespace)
    raw = api.put(np.stack((p/algebra["prefactor"], d/algebra["prefactor"]))[:, None])
    value, actual_slope = solve(raw, partner, 0, 1, object(), 0,
                                need_value=phase != "slope_only_resume")
    expected_full = base+r
    np.testing.assert_allclose(value, (base+dynamic)[None], atol=1e-13, rtol=2e-12)
    np.testing.assert_allclose(actual_slope, (expected_full@effective_d@expected_full+dynamic_slope)[None],
                               atol=1e-13, rtol=2e-12)
    if phase == "slope_only_resume":
        assert len(reads) == 1 and reads[0]["fields"] == ("Wc",)
        assert calls[-1][0] == "sample_slope" and len(calls[-1][1]) == 2
        np.testing.assert_allclose(calls[-1][1][0], expected_full[None], atol=1e-13, rtol=2e-12)
    else:
        assert reads == [] and calls[-1][0] == "sample_dyson" and len(calls[-1][1]) == 5

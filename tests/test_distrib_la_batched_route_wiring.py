"""Deck/CLI-to-service wiring for the universal distrib_la batch route."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_deck_route_defaults_normalizes_and_refuses_unknown(tmp_path):
    from gw.gw_config import (
        read_lorrax_input,
        resolve_distrib_la_batched_route,
    )

    bare = tmp_path / "bare.in"
    bare.write_text("[cohsex]\n")
    assert resolve_distrib_la_batched_route(
        read_lorrax_input(str(bare))) == "auto"

    opted = tmp_path / "opted.in"
    opted.write_text(
        "[cohsex]\n"
        "distrib_la_batched_route =  BATCH_RESHARD  \n")
    assert resolve_distrib_la_batched_route(
        read_lorrax_input(str(opted))) == "batch_reshard"

    with pytest.raises(ValueError, match="distrib_la_batched_route"):
        resolve_distrib_la_batched_route(
            {"distrib_la_batched_route": "replicate"})

    with pytest.raises(ValueError, match="contradicts use_low_mem_eigh"):
        resolve_distrib_la_batched_route({
            "distrib_la_batched_route": "batch_reshard",
            "use_low_mem_eigh": True,
        })


def test_deck_vocabulary_is_the_service_vocabulary():
    from ffi import _services
    _services.ensure_on_path()
    from distrib_la import BATCHED_ROUTE_CHOICES
    from gw.gw_config import distrib_la_batched_route_choices

    assert distrib_la_batched_route_choices() == tuple(BATCHED_ROUTE_CHOICES)
    assert tuple(BATCHED_ROUTE_CHOICES) == ("auto", "batch_reshard")


def test_every_direct_plan_batched_site_selects_at_plan_construction():
    """No production caller may reach the private ``_route=`` test seam."""
    found = []
    for rel in ("src/bandstructure/bse_setup.py", "src/bse/vq_interp.py",
                "src/gw/w_isdf.py", "src/isdf/core.py"):
        tree = ast.parse((ROOT / rel).read_text())
        for fn in (n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
            batched = [n for n in ast.walk(fn)
                       if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute)
                       and n.func.attr == "batched"]
            if not batched:
                continue
            plans = [n for n in ast.walk(fn)
                     if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name)
                     and n.func.id == "linalg_plan"]
            if plans:
                assert any(any(k.arg == "batched_route" for k in p.keywords)
                           for p in plans), f"{rel}:{fn.name} drops the route"
                found.extend((rel, fn.name) for _ in batched)
            for call in batched:
                assert all(k.arg != "_route" for k in call.keywords), (
                    f"{rel}:{fn.name} uses distrib_la's private gate override")

    assert len(found) == 6, f"Plan.batched inventory changed: {found}"


@pytest.mark.parametrize("rel", [
    "src/bandstructure/htransform.py",
    "src/bse/exciton_bands.py",
])
def test_cli_drivers_use_the_shared_deck_vocabulary(rel):
    src = (ROOT / rel).read_text()
    assert '"--distrib-la-batched-route"' in src
    assert "choices=distrib_la_batched_route_choices()" in src
    assert "resolve_distrib_la_batched_route(" in src


def test_exciton_cli_override_reaches_refit_htransform_calls():
    exciton = (ROOT / "src/bse/exciton_bands.py").read_text()
    refit = (ROOT / "src/bse/vq_interp.py").read_text()
    loading = (ROOT / "src/bse/bse_loading.py").read_text()
    densify = (ROOT / "src/bse/bse_densify.py").read_text()
    assert "distrib_la_batched_route=(\n" in exciton
    assert "override=distrib_la_batched_route" in refit
    assert '"distrib_la_batched_route": _distrib_la_batched_route' in refit
    assert "distrib_la_batched_route=distrib_la_batched_route" in loading
    assert "override=distrib_la_batched_route" in densify


def test_gwjax_routes_reach_zeta_w_and_qsgw_callers():
    checks = {
        "src/gw/gw_init.py": "cfg.backend.distrib_la_batched_route",
        "src/gw/screening.py": '"distrib_la_batched_route", "auto"',
        "src/gw/sc_iteration.py": '"distrib_la_batched_route", "auto"',
    }
    for rel, needle in checks.items():
        assert needle in (ROOT / rel).read_text(), f"{rel} drops {needle}"

    qsgw = (ROOT / "src/gw/qsgw_density.py").read_text()
    assert "batched_route=distrib_la_batched_route" in qsgw


def test_w_dyson_a_build_uses_top_level_distrib_la_matmul():
    """No explicit W/sqrt(P) JAX panels remain in the screening path."""
    source = (ROOT / "src/gw/w_isdf.py").read_text()
    tree = ast.parse(source)
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "linalg_matmul"
    ]
    assert calls, "W Dyson A-build no longer calls distrib_la.matmul"
    assert all(any(k.arg == "batched_route" for k in c.keywords)
               for c in calls)
    assert "V_row = jax.lax.all_gather" not in source
    assert "chi_col = jax.lax.all_gather" not in source
    assert "W Dyson A-build (distrib_la GEMM)" in source


def test_w_dyson_a_build_numerical_service_seam(monkeypatch):
    """Pin the real I-V(pref*chi), q-chunk, and solve handoff algebra."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from ffi import _services
    _services.ensure_on_path()
    import distrib_la as D
    import gw.w_isdf as W
    import isdf.core as IC

    jax.config.update("jax_enable_x64", True)
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1),
                ("x", "y"))
    face = NamedSharding(mesh, P(None, "x", "y"))
    calls = {"matmul": [], "plan": [], "solve": []}

    def fake_matmul(A, B, **kwargs):
        calls["matmul"].append((kwargs["backend"],
                                kwargs["batched_route"], int(A.shape[0])))
        return jnp.matmul(A, B)

    class FakePlan:
        backend = "scalapack"

        def describe(self):
            return "mock-scalapack-solve_lu"

        def batched(self, A, B):
            calls["solve"].append(
                (np.asarray(jax.device_get(A)), np.asarray(jax.device_get(B))))
            return jnp.linalg.solve(A, B)

    def fake_plan(op, mesh_arg, **kwargs):
        calls["plan"].append((op, mesh_arg, kwargs))
        return FakePlan()

    monkeypatch.setattr(W, "_w_solve_cache", {})
    monkeypatch.setattr(IC, "_chunk_q", lambda nq, per_q: 2)
    monkeypatch.setattr(IC, "_chunk_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(D, "matmul", fake_matmul)
    monkeypatch.setattr(D, "plan", fake_plan)

    rng = np.random.default_rng(9321)
    V_host = (rng.standard_normal((3, 4, 4))
              + 1j*rng.standard_normal((3, 4, 4)))
    chi_host = 0.02 * (rng.standard_normal((3, 4, 4))
                       + 1j*rng.standard_normal((3, 4, 4)))
    V = jax.device_put(V_host, face)
    chi = jax.device_put(chi_host, face)
    pref = 0.125
    fn = W._get_w_solve_fn_distributed(
        mesh, 3, 4, 4, distrib_la_batched_route="batch_reshard")
    got = np.asarray(jax.device_get(fn(V, chi, jnp.asarray(pref))))

    A_ref = np.eye(4)[None] - V_host @ (pref * chi_host)
    want = np.linalg.solve(A_ref, V_host)
    A_seen, B_seen = calls["solve"][0]
    np.testing.assert_allclose(A_seen, A_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(B_seen, V_host, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    assert calls["matmul"] == [
        ("scalapack", "batch_reshard", 2),
        ("scalapack", "batch_reshard", 1),
    ]
    assert calls["plan"] == [
        ("solve_lu", mesh, {
            "backend": "distributed", "n": 4,
            "batched_route": "batch_reshard",
        })
    ]


def test_explicit_route_does_not_take_native_bypass_branches():
    bse = (ROOT / "src/bandstructure/bse_setup.py").read_text()
    vq = (ROOT / "src/bse/vq_interp.py").read_text()
    expected = 'distrib_la_batched_route == "auto"'
    assert expected in bse
    assert expected in vq


def test_kmeans_is_preprocessing_not_an_inert_distrib_la_consumer():
    cli = (ROOT / "src/centroid/kmeans_cli.py").read_text()
    implementation = (ROOT / "src/centroid/pivoted_cholesky.py").read_text()
    assert "--distrib-la-batched-route" not in cli
    assert "linalg_plan(" not in implementation
    assert ".batched(" not in implementation

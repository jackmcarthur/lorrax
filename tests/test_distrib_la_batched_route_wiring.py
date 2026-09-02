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
        read_lorrax_input(str(bare))) == "batch_reshard"

    low_mem = tmp_path / "low_mem.in"
    low_mem.write_text("[cohsex]\nuse_low_mem_eigh = true\n")
    assert resolve_distrib_la_batched_route(
        read_lorrax_input(str(low_mem))) == "auto"

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
    from distrib_la import BATCHED_ROUTE_DEFAULT
    from gw.gw_config import (DISTRIB_LA_BATCHED_ROUTE_DEFAULT,
                              distrib_la_batched_route_choices)

    assert distrib_la_batched_route_choices() == tuple(BATCHED_ROUTE_CHOICES)
    assert tuple(BATCHED_ROUTE_CHOICES) == ("auto", "batch_reshard")
    assert BATCHED_ROUTE_DEFAULT == DISTRIB_LA_BATCHED_ROUTE_DEFAULT


def test_every_direct_plan_batched_site_selects_at_plan_construction():
    """No production caller may reach the private ``_route=`` test seam."""
    found = []
    for rel in ("src/bandstructure/bse_setup.py", "src/bse/vq_interp.py",
                "src/gw/w_isdf.py", "src/isdf/core.py",
                "src/isdf/galerkin.py"):
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

    assert len(found) == 7, f"Plan.batched inventory changed: {found}"


def test_htransform_galerkin_receives_the_one_resolved_run_route():
    """Every production door to the Galerkin batch uses the same setting."""
    checks = {
        "src/bandstructure/htransform.py": {
            "fit_galerkin_basis": "distrib_la_batched_route",
            "streaming_galerkin_solve": "distrib_la_batched_route",
            "initialize_wfns": "distrib_la_batched_route",
        },
        "src/bse/exciton_bands.py": {
            "initialize_wfns": "args.distrib_la_batched_route",
        },
        "src/bse/vq_interp.py": {
            "initialize_wfns": "_distrib_la_batched_route",
        },
        "src/bse/bse_densify.py": {
            "initialize_wfns": "_distrib_la_batched_route",
        },
    }
    for rel, expected in checks.items():
        tree = ast.parse((ROOT / rel).read_text())
        found = {}
        for call in (node for node in ast.walk(tree)
                     if isinstance(node, ast.Call)):
            name = (call.func.id if isinstance(call.func, ast.Name)
                    else call.func.attr
                    if isinstance(call.func, ast.Attribute) else None)
            if name not in expected:
                continue
            route = next((kw.value for kw in call.keywords
                          if kw.arg == "distrib_la_batched_route"), None)
            assert route is not None, f"{rel}:{name} drops the run route"
            found[name] = ast.unparse(route)
        assert found == expected


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
        "src/gw/screening.py": '"distrib_la_batched_route", "batch_reshard"',
        "src/gw/sc_iteration.py": '"distrib_la_batched_route", "batch_reshard"',
    }
    for rel, needle in checks.items():
        assert needle in (ROOT / rel).read_text(), f"{rel} drops {needle}"

    qsgw = (ROOT / "src/gw/qsgw_density.py").read_text()
    assert "batched_route=distrib_la_batched_route" in qsgw


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

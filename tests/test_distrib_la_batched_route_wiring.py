"""Deck/CLI-to-service wiring for the universal distrib_la batch route."""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


_PRODUCTION_GEMM_PLAN_INVENTORY = Counter({
    ("src/common/contract_bands.py", "_face_project_kernel"): 2,
    ("src/gw/cohsex_sigma.py", "_make_cohsex_kernels_face"): 1,
    ("src/gw/isdf_fitting.py", "fit_zeta_to_h5"): 1,
    ("src/gw/photon_sigma.py", "_make_photon_static_block_kernel"): 1,
    ("src/gw/ppm_sigma.py", "_face_g_plan"): 1,
    ("src/gw/ppm_tau_kernel.py", "_get_sigma_kij_kernel"): 2,
    ("src/gw/w_isdf.py", "_get_chi_minimax_kernel_face"): 1,
    ("src/gw/w_isdf.py", "_get_fused_photon_chi_kernel"): 1,
    ("src/gw/w_isdf.py", "_get_chi_fractional_contour_kernel_face"): 1,
    ("src/gw/w_isdf.py", "_get_finite_transfer_current_block_kernel"): 1,
    ("src/gw/wavefunction_bundle.py", "_face_rotate_kernel"): 2,
})


# Each factory below retains a callable that directly or transitively closes
# over one of the GemmPlans above.  Its cache key must therefore distinguish
# the run route; otherwise a later request can silently receive the executable
# built for an earlier route in the same process.
_ROUTE_KEYED_PLAN_FACTORIES = {
    "src/gw/cohsex_sigma.py": {
        "_make_cohsex_kernels": "cache_key",
    },
    "src/gw/photon_sigma.py": {
        "_make_photon_static_block_kernel": "key",
    },
    "src/gw/ppm_tau_kernel.py": {
        "get_sigma_spatial_kernel": "key",
        "_get_sigma_kij_kernel": "key",
        "_get_sigma_tau_kernel": "cache_key",
        "get_shared_sigma_tau_kernel": "key",
    },
    "src/gw/w_isdf.py": {
        "_get_chi_minimax_kernel": "cache_key",
        "_get_fused_photon_chi_kernel": "key",
        "_get_chi_fractional_contour_kernel": "cache_key",
        "_get_finite_transfer_current_block_kernel": "key",
    },
    "src/gw/wavefunction_bundle.py": {
        "_face_rotate_kernel": "key",
    },
}


# Public/orchestrator-to-factory seams that must carry the same resolved route
# all the way to the plan-owning functions above.  Values are exact call counts
# within the named owner; a duplicate/new path is intentionally a review event.
_TRANSITIVE_ROUTE_SEAMS = {
    "src/gw/sigma_dispatch.py": {
        "compute_sigma_xc": {
            "compute_static_photon_sigma": 1,
            "compute_cohsex_sigma": 1,
            "compute_sigma_x": 1,
            "compute_sigma_c_mpa_omega_grid": 1,
            "compute_ppm_sigma_pipeline": 1,
        },
    },
    "src/gw/photon_sigma.py": {
        "compute_static_photon_sigma": {
            "_make_photon_static_block_kernel": 1,
        },
        "_make_photon_static_block_kernel": {
            "contract_bands_block_reshard": 1,
        },
    },
    "src/gw/cohsex_sigma.py": {
        "compute_cohsex_sigma": {
            "_make_cohsex_kernels": 1,
            "compute_sigma_x_bispinor": 1,
        },
        "compute_sigma_x": {
            "_make_cohsex_kernels": 1,
            "compute_sigma_x_bispinor": 1,
        },
        "_make_cohsex_kernels": {
            "_make_cohsex_kernels_face": 1,
        },
        "_make_cohsex_kernels_face": {
            "contract_bands_block_reshard": 1,
        },
    },
    "src/gw/sigma_x_bispinor.py": {
        "compute_sigma_x_bispinor": {
            "_make_cohsex_kernels": 1,
        },
    },
    "src/gw/ppm_pipeline.py": {
        "compute_ppm_sigma_pipeline": {
            "precompile_sigma": 1,
            "compute_sigma_c_ppm_omega_grid": 1,
        },
    },
    "src/gw/ppm_tau_kernel.py": {
        "precompile_sigma": {"_get_sigma_tau_kernel": 2},
        "_get_sigma_tau_kernel": {"_get_sigma_kij_kernel": 1},
        "get_shared_sigma_tau_kernel": {"_get_sigma_kij_kernel": 1},
        "_get_sigma_kij_kernel": {"get_sigma_spatial_kernel": 1},
        "get_sigma_spatial_kernel": {
            "_make_project_ri_reduce_scatter": 1,
        },
        "_make_project_ri_reduce_scatter": {
            "contract_bands_block_reshard": 1,
        },
    },
    "src/gw/ppm_sigma.py": {
        "_run_sigma_branch": {"_get_sigma_tau_kernel": 2},
        "_compute_invalid_static_sigma": {
            "get_sigma_spatial_kernel": 1,
            "_face_g_plan": 1,
        },
        "_invalid_static_coh_by_bracket": {
            "get_sigma_spatial_kernel": 1,
            "_face_g_plan": 1,
        },
        "compute_sigma_c_ppm_omega_grid": {
            "_compute_invalid_static_sigma": 1,
            "_invalid_static_coh_by_bracket": 1,
        },
    },
    "src/gw/mpa/sigma.py": {
        "compute_sigma_c_mpa_omega_grid": {"integrate_sigma_store": 1},
        "integrate_sigma_store": {"_integrate_sigma_batches": 1},
        "_integrate_sigma_batches": {"get_shared_sigma_tau_kernel": 1},
    },
    "src/gw/sc_iteration.py": {
        "gw_iteration_map": {"rotate_wavefunctions": 1},
    },
    "src/gw/wavefunction_bundle.py": {
        "rotate_wavefunctions": {"_rotate_wavefunctions_face": 1},
        "_rotate_wavefunctions_face": {"_face_rotate_kernel": 1},
    },
    "src/gw/w_isdf.py": {
        "compute_static_photon_response": {
            "compute_experimental_no_pair_photon_chi0": 1,
        },
        "compute_experimental_no_pair_photon_chi0": {
            "_get_fused_photon_chi_kernel": 1,
        },
    },
}


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _function_args(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)
    return {arg.arg for arg in args}


def _gemm_plan_names(tree: ast.AST) -> set[str]:
    """Recognize direct imports and aliases such as ``_gemm_plan``."""
    names = {"gemm_plan"}  # also covers ``distrib_la.gemm_plan(...)``
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module != "distrib_la":
            continue
        for alias in node.names:
            if alias.name == "gemm_plan":
                names.add(alias.asname or alias.name)
    return names


def _function_by_name(tree: ast.AST, name: str):
    found = [node for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name == name]
    assert len(found) == 1, f"expected one production function {name}: {found}"
    return found[0]


def _assigned_value(fn: ast.AST, variable: str):
    found = []
    for node in ast.walk(fn):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == variable
               for target in targets):
            found.append(node.value)
    assert len(found) == 1, (
        f"{getattr(fn, 'name', '<function>')} must assign one {variable} "
        f"cache key, found {len(found)}")
    return found[0]


def _calls_named(fn: ast.AST, name: str) -> list[ast.Call]:
    return [node for node in ast.walk(fn)
            if isinstance(node, ast.Call) and _call_name(node) == name]


def _keyword_value(call: ast.Call, name: str):
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


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
        "src/gw/screening.py": '"distrib_la_batched_route", "auto"',
        "src/gw/sc_iteration.py": '"distrib_la_batched_route", "auto"',
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


def test_every_production_gemm_plan_explicitly_uses_the_run_route():
    """No reshardable planned GEMM may fall back to its own ``auto``.

    The exact inventory makes a newly added production GemmPlan a review
    event: it must either join this universal run dial or be documented and
    tested as structurally non-reshardable instead of silently escaping it.
    """
    inventory = Counter()
    for path in sorted((ROOT / "src").rglob("*.py")):
        tree = ast.parse(path.read_text())
        gemm_plan_names = _gemm_plan_names(tree)
        parents = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        for call in (node for node in ast.walk(tree)
                     if isinstance(node, ast.Call)
                     and _call_name(node) in gemm_plan_names):
            owner = call
            while owner in parents and not isinstance(
                    owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                owner = parents[owner]
            assert isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)), (
                f"{path.relative_to(ROOT)}:{call.lineno} gemm_plan is not "
                "owned by a production factory")
            rel = str(path.relative_to(ROOT))
            inventory[(rel, owner.name)] += 1

            assert "distrib_la_batched_route" in _function_args(owner), (
                f"{rel}:{owner.name} builds a GemmPlan but cannot receive "
                "the universal run route")
            route = next((kw.value for kw in call.keywords
                          if kw.arg == "batched_route"), None)
            assert route is not None, (
                f"{rel}:{call.lineno} gemm_plan drops batched_route=")
            route_names = {node.id for node in ast.walk(route)
                           if isinstance(node, ast.Name)}
            assert "distrib_la_batched_route" in route_names, (
                f"{rel}:{call.lineno} gemm_plan hard-codes or substitutes "
                f"the route: {ast.unparse(route)}")

    assert inventory == _PRODUCTION_GEMM_PLAN_INVENTORY, (
        "production gemm_plan inventory changed; classify the new/removed "
        f"site explicitly: got={inventory}")


def test_every_cached_gemm_plan_factory_keys_on_the_run_route():
    """A cached closure built for one route cannot serve another route."""
    for rel, factories in _ROUTE_KEYED_PLAN_FACTORIES.items():
        tree = ast.parse((ROOT / rel).read_text())
        for name, key_name in factories.items():
            fn = _function_by_name(tree, name)
            assert "distrib_la_batched_route" in _function_args(fn), (
                f"{rel}:{name} caches a GemmPlan closure but cannot receive "
                "the universal run route")
            key = _assigned_value(fn, key_name)
            key_names = {node.id for node in ast.walk(key)
                         if isinstance(node, ast.Name)}
            assert "distrib_la_batched_route" in key_names, (
                f"{rel}:{name} cache {key_name} omits the run route: "
                f"{ast.unparse(key)}")


def test_transitive_gemm_plan_callers_forward_the_run_route():
    """The one dispatch setting reaches every plan-owning factory chain."""
    for rel, owners in _TRANSITIVE_ROUTE_SEAMS.items():
        tree = ast.parse((ROOT / rel).read_text())
        for owner_name, expected in owners.items():
            owner = _function_by_name(tree, owner_name)
            is_dispatch = (rel, owner_name) == (
                "src/gw/sigma_dispatch.py", "compute_sigma_xc")
            is_sc_entry = (rel, owner_name) == (
                "src/gw/sc_iteration.py", "gw_iteration_map")

            if is_dispatch:
                resolved = _assigned_value(
                    owner, "distrib_la_batched_route")
                source = ast.unparse(resolved)
                assert "config.backend" in source
                assert "distrib_la_batched_route" in source
            elif not is_sc_entry:
                assert "distrib_la_batched_route" in _function_args(owner), (
                    f"{rel}:{owner_name} cannot receive the run route")

            for callee, count in expected.items():
                calls = _calls_named(owner, callee)
                assert len(calls) == count, (
                    f"{rel}:{owner_name} expected {count} call(s) to "
                    f"{callee}, found {len(calls)}")
                for call in calls:
                    route = _keyword_value(
                        call, "distrib_la_batched_route")
                    assert route is not None, (
                        f"{rel}:{call.lineno} {owner_name}->{callee} drops "
                        "distrib_la_batched_route=")
                    source = ast.unparse(route)
                    if is_sc_entry:
                        assert "inputs.config.backend" in source
                        assert "distrib_la_batched_route" in source
                    else:
                        route_names = {
                            node.id for node in ast.walk(route)
                            if isinstance(node, ast.Name)
                        }
                        assert "distrib_la_batched_route" in route_names, (
                            f"{rel}:{call.lineno} {owner_name}->{callee} "
                            f"substitutes route {source}")


def test_ppm_executor_branch_kwargs_carry_the_run_route():
    """The branch loop forwards its route through the one shared kwargs map."""
    tree = ast.parse((ROOT / "src/gw/ppm_sigma.py").read_text())
    owner = _function_by_name(tree, "compute_sigma_c_ppm_omega_grid")
    bundle = _assigned_value(owner, "common_branch_kwargs")
    assert isinstance(bundle, ast.Call) and _call_name(bundle) == "dict"
    route = _keyword_value(bundle, "distrib_la_batched_route")
    assert route is not None
    assert "distrib_la_batched_route" in {
        node.id for node in ast.walk(route) if isinstance(node, ast.Name)}

    calls = _calls_named(owner, "_run_sigma_branch")
    assert len(calls) == 1
    expansions = [kw.value for kw in calls[0].keywords if kw.arg is None]
    assert any(isinstance(value, ast.Name)
               and value.id == "common_branch_kwargs"
               for value in expansions), (
        "compute_sigma_c_ppm_omega_grid no longer expands its route-bearing "
        "common_branch_kwargs into _run_sigma_branch")

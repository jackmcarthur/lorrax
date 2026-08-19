"""Source-level pins for compatibility arguments retired from the ISDF path."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _function_signature(relative_path: str, name: str) -> tuple[set[str], str | None]:
    tree = ast.parse((ROOT / relative_path).read_text())
    function = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    args = function.args
    names = {
        arg.arg
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)
    }
    return names, args.kwarg.arg if args.kwarg is not None else None


def _top_level_names(relative_path: str, node_type) -> set[str]:
    tree = ast.parse((ROOT / relative_path).read_text())
    return {
        node.name for node in tree.body
        if isinstance(node, node_type)
    }


def _literal_dict_keys(relative_path: str, assignment: str) -> set[str]:
    tree = ast.parse((ROOT / relative_path).read_text())
    node = next(
        node for node in tree.body
        if ((isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name)
                     and target.id == assignment for target in node.targets))
            or (isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == assignment))
    )
    assert isinstance(node.value, ast.Dict)
    return {
        key.value for key in node.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def test_vq_dispatcher_has_no_retired_rspace_chunk_selectors():
    names, catch_all = _function_signature(
        "src/gw/compute_vcoul.py", "compute_all_V_q")
    assert {"mu_chunk_size", "q_batch_size"}.isdisjoint(names)
    assert catch_all is None


def test_isdf_entrypoint_does_not_swallow_retired_keywords():
    _, catch_all = _function_signature(
        "src/gw/gw_init.py", "prepare_isdf_and_wavefunctions")
    assert catch_all is None


def test_psi_g_store_has_no_ignored_symmetry_argument():
    names, catch_all = _function_signature(
        "src/common/psi_G_store.py", "build_psi_G_store")
    assert "sym" not in names
    assert catch_all is None


def test_redundant_gspace_lifecycle_is_retired():
    fit_names, _ = _function_signature(
        "src/gw/isdf_fitting.py", "fit_zeta_to_h5")
    store_names, _ = _function_signature(
        "src/common/psi_G_store.py", "build_psi_G_store")
    assert "gspace_mode" not in fit_names
    assert "mode" not in store_names
    assert "gspace_mode" not in _literal_dict_keys(
        "src/gw/gw_config.py", "_DEFAULTS")
    assert "gspace_mode" not in _literal_dict_keys(
        "tools/gen_input_reference.py", "KEYS")
    assert "GspaceIO" not in _top_level_names(
        "src/gw/gw_config.py", ast.ClassDef)
    assert "RereadPsiGStore" not in _top_level_names(
        "src/common/psi_G_store.py", ast.ClassDef)


def test_inert_zct_cap_is_retired():
    functions = _top_level_names("src/gw/gw_config.py", ast.FunctionDef)
    assert "resolve_zct_stage_cap" not in functions
    tree = ast.parse((ROOT / "src/gw/gw_config.py").read_text())
    memory_config = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MemoryConfig")
    fields = {
        node.target.id for node in memory_config.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
    }
    assert "zct_stage_cap_gb" not in fields

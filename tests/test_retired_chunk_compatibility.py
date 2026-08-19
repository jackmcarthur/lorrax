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

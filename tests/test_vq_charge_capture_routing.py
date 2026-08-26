"""Structural gate for the charge-tile pre-unfold capture owner.

The scalar and bispinor V orchestrators share one per-tile producer.  Human
display/dataset labels differ between them, so charge ownership must travel as
an explicit semantic field and never be reconstructed from either spelling.
"""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCER = ROOT / "src/gw/v_q_g_flat.py"
BISPINOR = ROOT / "src/gw/v_q_bispinor.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name)


def _calls(tree: ast.Module, name: str) -> list[ast.Call]:
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def _kw(call: ast.Call, name: str) -> ast.expr:
    return next(kw.value for kw in call.keywords if kw.arg == name)


def _if_owns_call(fn: ast.FunctionDef, owner: str, callee: str) -> bool:
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        if not isinstance(node.test, ast.Name) or node.test.id != owner:
            continue
        if any(
            isinstance(child, ast.Call)
            and ((isinstance(child.func, ast.Name)
                  and child.func.id == callee)
                 or (isinstance(child.func, ast.Attribute)
                     and child.func.attr == callee))
            for statement in node.body for child in ast.walk(statement)
        ):
            return True
    return False


def test_charge_capture_role_is_explicit_and_label_independent():
    producer_tree = _tree(PRODUCER)
    tile_fn = _function(producer_tree, "_compute_V_q_g_flat_one_tile")

    kwonly = {arg.arg: default for arg, default in zip(
        tile_fn.args.kwonlyargs, tile_fn.args.kw_defaults)}
    assert "is_charge_cc" in kwonly
    assert kwonly["is_charge_cc"] is None, "charge ownership must be required"

    for node in ast.walk(tile_fn):
        if not isinstance(node, ast.Compare):
            continue
        names = {
            child.id for child in ast.walk(node)
            if isinstance(child, ast.Name)
        }
        assert "timing_label" not in names, (
            "display text must not decide charge/capture semantics")

    assert _if_owns_call(
        tile_fn, "is_charge_cc", "measure_covariance")
    assert _if_owns_call(
        tile_fn, "is_charge_cc", "deposit_pre_unfold")


def test_scalar_and_bispinor_callers_supply_the_same_charge_semantics():
    producer_tree = _tree(PRODUCER)
    scalar_calls = _calls(producer_tree, "_compute_V_q_g_flat_one_tile")
    assert len(scalar_calls) == 1
    scalar_role = _kw(scalar_calls[0], "is_charge_cc")
    assert isinstance(scalar_role, ast.Constant) and scalar_role.value is True

    bispinor_tree = _tree(BISPINOR)
    bispinor_calls = _calls(bispinor_tree, "_compute_V_q_g_flat_one_tile")
    assert len(bispinor_calls) == 1
    bispinor_role = _kw(bispinor_calls[0], "is_charge_cc")
    assert isinstance(bispinor_role, ast.Name)
    assert bispinor_role.id == "is_CC"

    orchestrator = _function(
        bispinor_tree, "compute_V_q_bispinor_g_flat_to_h5")
    assignments = [
        node for node in ast.walk(orchestrator)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "is_CC"
                for target in node.targets)
    ]
    assert len(assignments) == 1
    assert ast.dump(assignments[0].value, include_attributes=False) == (
        "BoolOp(op=And(), values=[Compare(left=Name(id='mu_L', ctx=Load()), "
        "ops=[Eq()], comparators=[Constant(value=0)]), "
        "Compare(left=Name(id='nu_L', ctx=Load()), ops=[Eq()], "
        "comparators=[Constant(value=0)])])")

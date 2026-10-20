"""Static check: every call of a nested function supplies its required arguments.

A closure's call sites are not checked until the branch runs. 97fe1cad4 added
``room`` to ``produce_sample_bank``'s ``line_selector`` and missed the streamed
tier's remainder span, which raised a TypeError only on Fe 8^3 at 85 s. This test
reads every module under ``src`` and refuses a direct call of a nested ``def``
(one definition in its enclosing function, no ``*args``/``**kwargs`` on either
side) that leaves a required parameter unset.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _short_calls(tree):
    for outer in ast.walk(tree):
        if not isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        defs = {}
        for node in ast.walk(outer):
            if node is not outer and isinstance(node, ast.FunctionDef):
                defs.setdefault(node.name, []).append(node.args)
        for call in ast.walk(outer):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and len(defs.get(call.func.id, ())) == 1):
                continue
            a = defs[call.func.id][0]
            if (a.vararg or a.kwarg or any(isinstance(x, ast.Starred) for x in call.args)
                    or any(k.arg is None for k in call.keywords)):
                continue
            pos = a.posonlyargs + a.args
            given = {p.arg for p in pos[:len(call.args)]} | {k.arg for k in call.keywords}
            need = ({p.arg for p in pos[:len(pos) - len(a.defaults)]}
                    | {k.arg for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is None})
            if need - given or len(call.args) > len(pos):
                yield call.lineno, outer.name, call.func.id, sorted(need - given)


def test_nested_calls_supply_required_arguments():
    bad = [(str(p.relative_to(SRC.parent)), *row)
           for p in sorted(SRC.rglob("*.py")) for row in _short_calls(ast.parse(p.read_text()))]
    assert not bad, bad

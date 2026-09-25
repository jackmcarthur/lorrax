"""Every name the face COHSEX kernels read is bound, on every branch.

A branch no small deck executes can still hold a stale reference: after P2-C
deleted the dead ``wfns_g`` path, the (since deleted) Sigma_x spin-pair
stream still read ``wfns_g`` and every rank died with NameError at Sigma_x on
exactly the decks that reached it.  This cell resolves, statically, every
name loaded inside ``_make_cohsex_kernels_face`` and its nested kernels
against the Python scoping rules (parameters, locals, enclosing locals,
module globals, builtins), so a stale reference on a branch no test reaches
still fails here.
"""
import ast
import builtins
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "gw" / "cohsex_sigma.py"
FACTORY = "_make_cohsex_kernels_face"


def _bound_in(fn: ast.AST) -> set[str]:
    """Names a function binds: arguments, assignment/for/with/import targets, nested defs."""
    names = set()
    a = fn.args
    for arg in a.posonlyargs + a.args + a.kwonlyargs:
        names.add(arg.arg)
    for arg in (a.vararg, a.kwarg):
        if arg is not None:
            names.add(arg.arg)
    for node in ast.walk(fn):
        if node is fn:
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)          # lambda / nested-def arguments
    return names


def test_face_cohsex_kernels_read_only_bound_names():
    tree = ast.parse(SRC.read_text())
    module_names = _bound_in(ast.parse("def _m(): pass").body[0])
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                module_names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        module_names.add(n.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            module_names.add(node.target.id)
    factory = next(n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == FACTORY)
    # every name bound anywhere in the factory (its own and its kernels') is in
    # scope for its kernels: closures read enclosing locals at call time.
    in_scope = module_names | _bound_in(factory) | set(dir(builtins))
    unbound = sorted({n.id for n in ast.walk(factory)
                      if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                      and n.id not in in_scope})
    assert not unbound, (
        f"{FACTORY} reads names bound nowhere in scope: {unbound} "
        f"(a stale reference on a branch such as the spin-pair stream)")

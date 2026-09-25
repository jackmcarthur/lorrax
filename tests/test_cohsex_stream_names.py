"""Every name the face COHSEX kernels read is bound (the spin-pair stream branch included).

The Sigma_x / COH spin-pair stream (``stream``) engages only when the
whole-spin G_occ exceeds the device target (2 G_tile), e.g. the CrI3 6x6
mu3088 bispinor on 40 GB, so no small deck executes it.  After P2-C deleted
the dead ``wfns_g`` path, ``sigma_sx`` still read ``if stream and wfns_g is
None`` and every rank died with NameError at Sigma_x on exactly those decks.
This cell resolves, statically, every name loaded inside
``_make_cohsex_kernels_face`` and its nested kernels against the Python
scoping rules (parameters, locals, enclosing locals, module globals,
builtins), so a stale reference on a branch no test reaches still fails here.
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


def test_the_stream_branch_is_guarded_by_stream_alone():
    """sigma_sx and sigma_coh take the pair stream exactly when ``stream`` is set."""
    tree = ast.parse(SRC.read_text())
    factory = next(n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == FACTORY)
    kernels = {n.name: n for n in ast.walk(factory)
               if isinstance(n, ast.FunctionDef) and n.name in ("sigma_sx", "sigma_coh")}
    assert set(kernels) == {"sigma_sx", "sigma_coh"}
    for name, fn in kernels.items():
        guards = [n.test for n in ast.walk(fn) if isinstance(n, ast.If)
                  and any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == "_pair_sigma"
                          for s in n.body for c in ast.walk(s))]
        assert len(guards) == 1, name
        assert isinstance(guards[0], ast.Name) and guards[0].id == "stream", (
            name, ast.unparse(guards[0]))

#!/usr/bin/env python3
"""Audit every environment-variable read under a source tree.

    python3 tools/env_audit.py src

Walks the AST (not grep) for ``os.environ.get`` / ``os.getenv`` /
``os.environ[...]`` / ``os.environ.setdefault`` / ``"X" in os.environ`` and
prints each variable with its literal default at every site.  Sites whose
literal defaults DISAGREE are flagged ``<<< MULTIPLE DEFAULTS`` — that is
the drift this exists to catch (one module defaulting a knob to 4 while
another defaults it to 8 is invisible until it bites).

C++ ``getenv()`` reads are NOT covered; find those with
``grep -rn 'getenv(' src/ffi``.

Keep ``docs/dev/env_vars.md`` in sync with this tool's output.
"""
import ast, sys, collections
from pathlib import Path

ROOT = Path(sys.argv[1])
hits = collections.defaultdict(list)   # name -> [(file, line, default_repr)]


def lit(node):
    try:
        return repr(ast.literal_eval(node))
    except Exception:
        return "<expr>"


class V(ast.NodeVisitor):
    def __init__(self, path):
        self.path = path

    def visit_Call(self, node):
        f = node.func
        name = default = None
        # os.environ.get("X", d) / os.getenv("X", d)
        if isinstance(f, ast.Attribute) and f.attr in ("get", "getenv"):
            base = f.value
            ok = False
            if f.attr == "getenv" and isinstance(base, ast.Name) and base.id == "os":
                ok = True
            if f.attr == "get" and isinstance(base, ast.Attribute) and base.attr == "environ":
                ok = True
            if ok and node.args:
                a0 = node.args[0]
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    name = a0.value
                    default = lit(node.args[1]) if len(node.args) > 1 else "None"
        # os.environ.setdefault("X", d)
        if isinstance(f, ast.Attribute) and f.attr == "setdefault" and \
           isinstance(f.value, ast.Attribute) and f.value.attr == "environ" and node.args:
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                name = a0.value
                default = "setdefault " + (lit(node.args[1]) if len(node.args) > 1 else "?")
        if name:
            hits[name].append((str(self.path), node.lineno, default))
        self.generic_visit(node)

    def visit_Subscript(self, node):
        v = node.value
        if isinstance(v, ast.Attribute) and v.attr == "environ":
            s = node.slice
            if isinstance(s, ast.Constant) and isinstance(s.value, str):
                hits[s.value].append((str(self.path), node.lineno, "<required/assign>"))
        self.generic_visit(node)

    def visit_Compare(self, node):
        # "X" in os.environ
        if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
            for op, cmp in zip(node.ops, node.comparators):
                if isinstance(op, ast.In) and isinstance(cmp, ast.Attribute) \
                        and cmp.attr == "environ":
                    hits[node.left.value].append(
                        (str(self.path), node.lineno, "<presence test>"))
        self.generic_visit(node)


for py in sorted(ROOT.rglob("*.py")):
    try:
        V(py.relative_to(ROOT.parent)).visit(ast.parse(py.read_text()))
    except SyntaxError as e:
        print(f"!! parse fail {py}: {e}", file=sys.stderr)

for name in sorted(hits):
    sites = hits[name]
    defaults = sorted({d for _, _, d in sites})
    flag = "  <<< MULTIPLE DEFAULTS" if len([d for d in defaults
           if d not in ("<presence test>", "<required/assign>")]) > 1 else ""
    print(f"{name}  [{len(sites)} site(s)] defaults={defaults}{flag}")
    for f, l, d in sites:
        print(f"      {f}:{l}   {d}")

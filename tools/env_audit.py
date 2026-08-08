#!/usr/bin/env python3
"""Audit every environment-variable read under a source tree.

    python3 tools/env_audit.py src

Walks the AST (not grep) for ``os.environ.get`` / ``os.getenv`` /
``os.environ[...]`` / ``os.environ.setdefault`` / ``"X" in os.environ`` and
prints each variable with its literal default at every site.  Sites whose
literal defaults DISAGREE are flagged ``<<< MULTIPLE DEFAULTS`` — that is
the drift this exists to catch (one module defaulting a knob to 4 while
another defaults it to 8 is invisible until it bites).

HELPER-MEDIATED READS (the majority since the fix/zq grammar work) are
covered too: a call ``env_bool("LORRAX_X", ...)`` or ``Gate(env="LORRAX_X",
...)`` is an environment read exactly as much as ``os.environ.get`` is, and
the 2026-07 sweeps that funnelled raw reads through those helpers made the
raw-read-only version of this tool blind to most of the tree — a
FALSE-CLEAN sweep.  The recognised helper names are in ``ENV_HELPERS``;
keep that set in sync with the grammar helpers the tree actually defines
(``tests/test_env_registry.py`` shares the same visitor and has the
negative controls).

OLD-INTERPRETER HONESTY: on Python < 3.8 string literals parse as
``ast.Str``, not ``ast.Constant``.  The pre-2026-07-31 version tested only
``ast.Constant`` and therefore printed NOTHING on the Frontera login
node's python3 (3.7.0) — rc=0, empty report, indistinguishable from a
tree with no env reads.  ``const_str`` below accepts both spellings, and
the ``--selftest`` flag proves the walker sees a known fixture (run it if
you doubt the interpreter).

C++ ``getenv()`` reads are NOT covered by the AST walk; find those with
``grep -rn 'getenv(' src/ffi`` (``tests/test_env_registry.py`` scans them
by regex and enforces registry rows for both sides).

Keep ``docs/dev/env_vars.md`` in sync with this tool's output.
"""
import ast
import collections
import sys
from pathlib import Path

#: Helper callables whose FIRST positional string argument is an env-var
#: name.  ``while it exists``: ``_env_bool`` stays listed until the last
#: definition leaves the tree — an extra name here costs nothing, a missing
#: one re-arms the false-clean sweep.
ENV_HELPERS = frozenset({
    "env_bool", "env_float",              # gw.gw_config (canonical grammar)
    "_env_bool",                          # isdf.core (historical; alias now)
    "_env_falsy",                         # runtime.__init__ (two-valued)
    "_env_flag",                          # file_io._slab_io_mpi_host
    "_env_int",                           # symmetry_maps.density_symmetry_check
    "_env_float",                         # symmetry_maps.density_symmetry_check
    "_env_override_raw",                  # isdf.core (deprecated env twins)
    "_deprecated_env_float",              # isdf.core (deprecated env twins)
})

#: Callables whose ``env=`` keyword names an env var (typed capability
#: gates: ``ffi/gate.py``).
ENV_KWARG_CALLS = frozenset({"Gate"})


def const_str(node):
    """The str a literal node carries, or None.  Handles BOTH the modern
    ``ast.Constant`` and the pre-3.8 ``ast.Str`` spelling (see module
    docstring: testing only one of them made this tool a silent no-op on
    python 3.7)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    # Python < 3.8 (and deprecation-era trees): string literals are ast.Str.
    if hasattr(ast, "Str") and isinstance(node, ast.Str):
        return node.s
    return None


def lit(node):
    try:
        return repr(ast.literal_eval(node))
    except Exception:
        return "<expr>"


class EnvReadVisitor(ast.NodeVisitor):
    """Collects ``(name, lineno, default_repr)`` env-read sites."""

    def __init__(self, path, hits):
        self.path = path
        self.hits = hits

    def _add(self, name, lineno, default):
        self.hits[name].append((str(self.path), lineno, default))

    def visit_Call(self, node):
        f = node.func
        fname = f.id if isinstance(f, ast.Name) else (
            f.attr if isinstance(f, ast.Attribute) else None)
        # os.environ.get("X", d) / os.getenv("X", d)
        if isinstance(f, ast.Attribute) and f.attr in ("get", "getenv"):
            base = f.value
            ok = False
            if f.attr == "getenv" and isinstance(base, ast.Name) \
                    and base.id == "os":
                ok = True
            if f.attr == "get" and isinstance(base, ast.Attribute) \
                    and base.attr == "environ":
                ok = True
            if ok and node.args:
                name = const_str(node.args[0])
                if name:
                    default = (lit(node.args[1])
                               if len(node.args) > 1 else "None")
                    self._add(name, node.lineno, default)
        # os.environ.setdefault("X", d)
        elif isinstance(f, ast.Attribute) and f.attr == "setdefault" and \
                isinstance(f.value, ast.Attribute) and \
                f.value.attr == "environ" and node.args:
            name = const_str(node.args[0])
            if name:
                self._add(name, node.lineno, "setdefault " + (
                    lit(node.args[1]) if len(node.args) > 1 else "?"))
        # env_bool("X", d) / _env_falsy("X") / ... — helper-mediated reads
        elif fname in ENV_HELPERS and node.args:
            name = const_str(node.args[0])
            if name:
                default = (lit(node.args[1])
                           if len(node.args) > 1 else "<helper>")
                self._add(name, node.lineno,
                          "%s %s" % (fname, default))
        # Gate(env="X", ...) — typed capability gates
        if fname in ENV_KWARG_CALLS:
            for kw in node.keywords:
                if kw.arg == "env":
                    name = const_str(kw.value)
                    if name:
                        self._add(name, node.lineno, "Gate(env=...)")
        self.generic_visit(node)

    def visit_Subscript(self, node):
        v = node.value
        if isinstance(v, ast.Attribute) and v.attr == "environ":
            s = node.slice
            # py<3.9 wraps the subscript in ast.Index
            if s.__class__.__name__ == "Index":
                s = s.value
            name = const_str(s)
            if name:
                self._add(name, node.lineno, "<required/assign>")
        self.generic_visit(node)

    def visit_Compare(self, node):
        # "X" in os.environ
        name = const_str(node.left)
        if name is not None:
            for op, cmp in zip(node.ops, node.comparators):
                if isinstance(op, ast.In) and isinstance(cmp, ast.Attribute) \
                        and cmp.attr == "environ":
                    self._add(name, node.lineno, "<presence test>")
        self.generic_visit(node)


def collect(root):
    """name -> [(file, line, default_repr)] over every .py under ``root``."""
    hits = collections.defaultdict(list)
    for py in sorted(Path(root).rglob("*.py")):
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError as e:
            print("!! parse fail %s: %s" % (py, e), file=sys.stderr)
            continue
        EnvReadVisitor(py.relative_to(Path(root).parent), hits).visit(tree)
    return hits


_SELFTEST_SRC = '''
import os
a = os.environ.get("SELFTEST_RAW", "1")
b = os.getenv("SELFTEST_GETENV")
c = os.environ["SELFTEST_REQ"]
os.environ.setdefault("SELFTEST_SET", "x")
d = "SELFTEST_PRESENCE" in os.environ
e = env_bool("SELFTEST_HELPER", True)
f = _env_falsy("SELFTEST_FALSY")
g = Gate(env="SELFTEST_GATE", target="t", platforms=(), modes=(),
         default="off", off_label="")
'''

_SELFTEST_EXPECT = {
    "SELFTEST_RAW", "SELFTEST_GETENV", "SELFTEST_REQ", "SELFTEST_SET",
    "SELFTEST_PRESENCE", "SELFTEST_HELPER", "SELFTEST_FALSY",
    "SELFTEST_GATE",
}


def selftest():
    """Prove the walker sees every read shape ON THIS INTERPRETER.

    The false-clean failure mode this guards: an interpreter whose AST
    spells literals differently (3.7 ast.Str) walks the whole tree, matches
    nothing, and exits 0.  A tool that can silently see nothing must carry
    the proof that it can see something.
    """
    hits = collections.defaultdict(list)
    EnvReadVisitor(Path("<selftest>"), hits).visit(ast.parse(_SELFTEST_SRC))
    missing = _SELFTEST_EXPECT - set(hits)
    if missing:
        print("env_audit SELFTEST FAILED on python %s: the walker missed %s"
              % (sys.version.split()[0], sorted(missing)), file=sys.stderr)
        return 1
    return 0


def main(argv):
    if "--selftest" in argv:
        rc = selftest()
        print("selftest %s (python %s)"
              % ("ok" if rc == 0 else "FAILED", sys.version.split()[0]))
        return rc
    if selftest() != 0:            # never emit a report the walker can't back
        return 1
    root = Path(argv[1]) if len(argv) > 1 else Path("src")
    hits = collect(root)
    if not hits:
        print("!! no env reads found under %s — on a LORRAX tree that is "
              "always wrong; suspect the walker" % root, file=sys.stderr)
        return 1
    for name in sorted(hits):
        sites = hits[name]
        defaults = sorted({d for _, _, d in sites})
        flag = "  <<< MULTIPLE DEFAULTS" if len(
            [d for d in defaults
             if d not in ("<presence test>", "<required/assign>")
             and not d.startswith(("setdefault", "Gate("))]) > 1 else ""
        print("%s  [%d site(s)] defaults=%s%s"
              % (name, len(sites), defaults, flag))
        for f, l, d in sites:
            print("      %s:%d   %s" % (f, l, d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

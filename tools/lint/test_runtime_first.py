"""Lint: ``runtime`` is imported (and tunes OpenBLAS) before numpy/scipy/jax/h5py.

OpenBLAS reads its spin setting when its library is mapped, so ``runtime.blas_env.tune_at_import``
must run before anything maps numpy.  Three source-order rules keep that true (otherwise GATE
blas_spin refuses the run, ``runtime/blas_env.py``):

* the first statements of ``runtime/__init__.py`` import and call ``blas_env.tune_at_import``
  and ``blas_env`` itself imports only ``os``/``sys``/``ctypes``;
* every driver package (``gw``, ``bse``, ``centroid``, ``bandstructure``) imports ``runtime``
  first in its ``__init__``;
* a module that starts the runtime (``initialize_communicator_stack(``) imports ``runtime`` or a
  driver package before numpy/scipy/jax/h5py at module top.

Pure AST -- no jax, no src imports; runs on a login node
(``python3 tools/lint/test_runtime_first.py``) and under pytest.  The behavioural import-order
cells (a fresh interpreter each) stay in ``tests/test_runtime_blas_env.py``.
"""
import ast
import os

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_REPO, "src")

_DRIVER_PACKAGES = ("gw", "bse", "centroid", "bandstructure")
_BLAS_MODULES = ("numpy", "scipy", "jax", "h5py")


def test_runtime_tunes_before_any_other_import():
    """The first non-docstring, non-__future__, non-``os`` statements of runtime/__init__.py
    import and call blas_env.tune_at_import: nothing above them can map numpy."""
    body = ast.parse(open(os.path.join(_SRC, "runtime", "__init__.py")).read()).body
    body = [n for n in body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    head = [ast.unparse(n) for n in body[:5]]
    assert head[0] == "from __future__ import annotations" and head[1] == "import os", head
    assert "blas_env" in head[2] and "blas_env" in head[3] and head[4] == "_tune_blas_at_import()", head
    src = open(os.path.join(_SRC, "runtime", "blas_env.py")).read()
    mods = {a.name.split(".")[0] for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import) for a in n.names}
    assert mods <= {"os", "sys", "ctypes"}, mods


def test_driver_packages_import_runtime_first():
    for pkg in _DRIVER_PACKAGES:
        body = ast.parse(open(os.path.join(_SRC, pkg, "__init__.py")).read()).body
        imports = [n for n in body if isinstance(n, (ast.Import, ast.ImportFrom))]
        assert imports and ast.unparse(imports[0]) == "import runtime", pkg


def _first_relevant_import(tree):
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [node.module or ""]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            if top in ("runtime",) + _DRIVER_PACKAGES + _BLAS_MODULES:
                return top
    return None


def test_every_startup_caller_imports_runtime_before_numpy():
    """A module that starts the runtime (initialize_communicator_stack) imports runtime, or a driver
    package whose __init__ does, before numpy/scipy/jax/h5py at module top -- otherwise GATE
    blas_spin refuses it.  Modules inside the driver packages are covered by the package __init__."""
    roots = [os.path.join(_REPO, d) for d in ("src", "tests", "tools")]
    roots += [os.path.join(_REPO, "services", s, "bench") for s in os.listdir(os.path.join(_REPO, "services"))]
    bad = []
    for root in roots:
        for dirpath, _, files in os.walk(root):
            rel = os.path.relpath(dirpath, _SRC)
            if rel.split(os.sep)[0] in _DRIVER_PACKAGES + ("runtime",):
                continue
            for name in files:
                path = os.path.join(dirpath, name)
                if not name.endswith(".py") or path == os.path.abspath(__file__):
                    continue
                text = open(path, encoding="utf-8", errors="replace").read()
                if "initialize_communicator_stack(" not in text:
                    continue
                first = _first_relevant_import(ast.parse(text))
                if first in _BLAS_MODULES:
                    bad.append(f"{os.path.relpath(path, _REPO)} imports {first} first")
    assert not bad, bad


def _main():
    n_run, failures = 0, []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        n_run += 1
        try:
            fn()
        except Exception as exc:            # noqa: BLE001 — this IS the tally
            msg = (str(exc).splitlines() or [""])[0]
            failures.append("FAIL %s — %s: %s" % (name, type(exc).__name__, msg))
            print(failures[-1])
        else:
            print("ok   %s" % name)
    print("\n%d/%d passed, %d failed"
          % (n_run - len(failures), n_run, len(failures)))
    for line in failures:
        print(line)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())

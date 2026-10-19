"""OpenBLAS spin tuning (runtime.blas_env): set before numpy, witnessed per library, gated.

Import order is per process, so the order cells run a fresh interpreter each.  CPU only, no jax.
Run: ``PYTHONPATH=src:services/lxkit/src python3 -m pytest tests/test_runtime_blas_env.py``.
"""
import ast
import json
import os
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from runtime import blas_env                                       # noqa: E402

_CENSUS = ("import json, runtime.blas_env as b\n"
           "try:\n    b.enforce(); gate = None\n"
           "except RuntimeError as e:\n    gate = str(e)\n"
           "print(json.dumps({'libs': b.STATE['libs'], 'gate': gate, 'env': __import__('os').environ.get(b.VAR)}))")


def _run(prelude, env_extra=None, drop=("OPENBLAS_THREAD_TIMEOUT", "LORRAX_BLAS_TUNE")):
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env["PYTHONPATH"] = os.pathsep.join([_SRC, env.get("PYTHONPATH", "")])
    env.update(env_extra or {})
    out = subprocess.run([sys.executable, "-c", prelude + "\n" + _CENSUS], env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _need_openblas():
    try:
        import numpy  # noqa: F401
        import scipy.linalg  # noqa: F401
    except ImportError:
        pytest.skip("numpy/scipy not installed")
    if blas_env.mapped_openblas() is None:
        pytest.skip("no /proc/self/maps")
    if not blas_env.mapped_openblas():
        pytest.skip("numpy/scipy here are not linked to OpenBLAS")


def _states(res):
    return {r["state"] for r in res["libs"]}


# ---- the decision table (pure) ----------------------------------------------

@pytest.mark.parametrize("kw, want", [
    (dict(live=1, preloaded=True, caller_value=None, initial_value=None, initial_known=True, env_value="1"), "tuned"),
    (dict(live=0, preloaded=False, caller_value=None, initial_value=None, initial_known=True, env_value=None), "untuned"),
    (dict(live=0, preloaded=False, caller_value="0", initial_value="0", initial_known=True, env_value="0"), "caller"),
    (dict(live=28, preloaded=False, caller_value="28", initial_value="28", initial_known=True, env_value="28"), "caller"),
    (dict(live=None, preloaded=True, caller_value=None, initial_value=None, initial_known=True, env_value="1"), "untuned"),
    (dict(live=None, preloaded=True, caller_value=None, initial_value="1", initial_known=True, env_value="1"), "tuned"),
    (dict(live=None, preloaded=True, caller_value="1", initial_value=None, initial_known=True, env_value="1"), "caller"),
    (dict(live=None, preloaded=True, caller_value=None, initial_value=None, initial_known=False, env_value="1"), "unknown"),
    (dict(live=None, preloaded=False, caller_value=None, initial_value=None, initial_known=True, env_value="1"), "tuned"),
    (dict(live=None, preloaded=False, caller_value="30", initial_value="30", initial_known=True, env_value="30"), "caller"),
])
def test_classify_decision_table(kw, want):
    assert blas_env.classify("libx.so", **kw)[0] == want


# ---- import order, one process each ------------------------------------------

def test_runtime_first_tunes_every_copy():
    _need_openblas()
    res = _run("import runtime, numpy, scipy.linalg")
    assert res["gate"] is None and res["env"] == "1"
    assert _states(res) == {"tuned"} and len(res["libs"]) >= 1


def test_a_driver_package_first_tunes_numpy_too():
    """The Fe harnesses' order: ``import gw.shared_pole_recipe`` (numpy) before runtime."""
    _need_openblas()
    res = _run("import gw.shared_pole_recipe, scipy.linalg")
    assert res["gate"] is None and _states(res) == {"tuned"}


def test_numpy_first_still_tunes_scipy_and_the_gate_names_numpy():
    """The bug fixed 2026-10-15: numpy first used to skip the setdefault, leaving scipy's copy
    (the rule builders' LAPACK) untuned too.  Now scipy's reads it; numpy's is refused by name."""
    _need_openblas()
    res = _run("import numpy, runtime, scipy.linalg")
    assert res["env"] == "1"
    # Classified by library, not by witness: numpy's wheel may export the live symbol (it does
    # in the Perlmutter cuda13 venv), so its untuned row can carry witness 'live symbol' too.
    numpy_copy = [r for r in res["libs"] if "openblas64_" in r["lib"]]
    scipy_copy = [r for r in res["libs"] if "openblas64_" not in r["lib"]]
    assert scipy_copy and all(r["state"] == "tuned" for r in scipy_copy)
    assert numpy_copy and all(r["state"] == "untuned" for r in numpy_copy)
    assert res["gate"] and res["gate"].startswith("GATE blas_spin:")
    assert all(r["lib"] in res["gate"] for r in numpy_copy)


def test_an_exported_value_is_honoured_whatever_the_order():
    _need_openblas()
    res = _run("import numpy, runtime, scipy.linalg", {"OPENBLAS_THREAD_TIMEOUT": "1"})
    assert res["gate"] is None and _states(res) == {"tuned"}
    res = _run("import numpy, runtime, scipy.linalg", {"OPENBLAS_THREAD_TIMEOUT": "28"})
    assert res["gate"] is None and _states(res) == {"caller"}


def test_the_opt_out_declines_tuning_and_gate():
    _need_openblas()
    res = _run("import numpy, runtime, scipy.linalg", {"LORRAX_BLAS_TUNE": "0"})
    assert res["gate"] is None and res["env"] is None


# ---- the order, held in source -------------------------------------------------

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


@pytest.mark.parametrize("pkg", ["gw", "bse", "centroid", "bandstructure"])
def test_driver_packages_import_runtime_first(pkg):
    body = ast.parse(open(os.path.join(_SRC, pkg, "__init__.py")).read()).body
    imports = [n for n in body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert imports and ast.unparse(imports[0]) == "import runtime", pkg


_DRIVER_PACKAGES = ("gw", "bse", "centroid", "bandstructure")
_BLAS_MODULES = ("numpy", "scipy", "jax", "h5py")


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

"""The BLAS spin setting only works if ``runtime`` is imported first.

``runtime.tune_blas_threading`` sets ``OPENBLAS_THREAD_TIMEOUT`` at import of
the runtime module.  OpenBLAS reads that variable ONCE, in the constructor
that runs at the first ``import numpy``, so an entry point that imports numpy
(or anything pulling it in: jax, scipy, h5py) before ``runtime`` gets the
library default and its host LAPACK work runs the slow way -- measured 5.2x
on the Sigma box-rule planner.  Nothing about that failure is visible in a
result, only in the wall, which is exactly the kind of rot a test has to hold.
"""
import ast
import os
import pathlib
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"

#: importing any of these executes ``import numpy`` somewhere underneath, so
#: OpenBLAS initialises and the setting is already too late.
PULLS_IN_NUMPY = {"numpy", "jax", "scipy", "h5py", "pandas", "matplotlib"}


#: a module that CALLS one of these starts a run, so its own import order is
#: what decides whether the BLAS setting lands.
STARTUP_CALLS = {"initialize_communicator_stack", "run_main_and_finalize"}


def _calls_the_startup(tree):
    """True when the module really calls the startup -- parsed, not grepped,
    so a mention in a comment or docstring does not count (``vq_interp.py``
    names it in a comment and is not an entry point)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute) else None)
        if name in STARTUP_CALLS:
            return True
    return False


def _entry_points():
    """Modules that start a run: they call the startup themselves."""
    out = []
    for path in sorted(SRC.glob("*/*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"),
                             str(path))
        except SyntaxError:                      # not ours to police
            continue
        if _calls_the_startup(tree):
            out.append(path)
    return out


def _first_import_lines(path):
    """(first runtime import line, first numpy-pulling import line, name)."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), str(path))
    runtime_line = numpy_line = None
    numpy_name = None
    for node in ast.walk(tree):
        roots = []
        if isinstance(node, ast.Import):
            roots = [(a.name.split(".")[0], node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots = [(node.module.split(".")[0], node.lineno)]
        for root, lineno in roots:
            if root == "runtime" and (runtime_line is None or lineno < runtime_line):
                runtime_line = lineno
            if root in PULLS_IN_NUMPY and (numpy_line is None or lineno < numpy_line):
                numpy_line, numpy_name = lineno, root
    return runtime_line, numpy_line, numpy_name


@pytest.mark.parametrize("path", _entry_points(), ids=lambda p: p.name)
def test_entry_point_imports_runtime_before_numpy(path):
    runtime_line, numpy_line, numpy_name = _first_import_lines(path)
    if runtime_line is None or numpy_line is None:
        pytest.skip(f"{path.name} does not import both runtime and numpy")
    assert runtime_line < numpy_line, (
        f"{path.relative_to(SRC.parent)} imports {numpy_name} at line "
        f"{numpy_line} before runtime at line {runtime_line}. OpenBLAS reads "
        f"OPENBLAS_THREAD_TIMEOUT in the constructor that runs with numpy, so "
        f"runtime.tune_blas_threading comes too late and this driver's host "
        f"LAPACK work runs the slow way. Move the runtime import above it.")


def test_entry_points_were_actually_found():
    """A silent empty parametrisation would make the test above vacuous."""
    assert len(_entry_points()) >= 4


def test_sets_the_timeout_when_numpy_is_not_yet_imported(monkeypatch):
    import runtime
    monkeypatch.delenv("OPENBLAS_THREAD_TIMEOUT", raising=False)
    monkeypatch.delenv("LORRAX_BLAS_TUNE", raising=False)
    monkeypatch.delitem(sys.modules, "numpy", raising=False)
    assert runtime.tune_blas_threading() is True
    assert os.environ["OPENBLAS_THREAD_TIMEOUT"] == "1"


def test_caller_override_wins(monkeypatch):
    import runtime
    monkeypatch.setenv("OPENBLAS_THREAD_TIMEOUT", "26")
    monkeypatch.delenv("LORRAX_BLAS_TUNE", raising=False)
    monkeypatch.delitem(sys.modules, "numpy", raising=False)
    assert runtime.tune_blas_threading() is True
    assert os.environ["OPENBLAS_THREAD_TIMEOUT"] == "26"


def test_opt_out(monkeypatch):
    import runtime
    monkeypatch.setenv("LORRAX_BLAS_TUNE", "0")
    monkeypatch.delenv("OPENBLAS_THREAD_TIMEOUT", raising=False)
    monkeypatch.delitem(sys.modules, "numpy", raising=False)
    assert runtime.tune_blas_threading() is False
    assert "OPENBLAS_THREAD_TIMEOUT" not in os.environ


def test_says_so_when_numpy_got_there_first(monkeypatch):
    """The one failure that must never be silent: the setting is too late."""
    import numpy  # noqa: F401  - deliberately present in sys.modules
    import runtime
    monkeypatch.delenv("LORRAX_BLAS_TUNE", raising=False)
    before = len(runtime._DEMOTIONS)
    assert runtime.tune_blas_threading() is False
    assert runtime._BLAS_TUNE["numpy_already_imported"] is True
    assert "numpy" in (runtime._BLAS_TUNE["reason"] or "")
    assert len(runtime._DEMOTIONS) > before


def test_startup_report_wires_the_state():
    """``collect_startup_facts`` itself needs a live backend, so check the
    wiring rather than calling it: the report must carry ``blas_tune`` or the
    run cannot say whether the tuning was armed."""
    src = (SRC / "runtime" / "__init__.py").read_text(encoding="utf-8")
    assert 'f["blas_tune"] = dict(_BLAS_TUNE)' in src

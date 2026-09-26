"""Standalone, falsifiably — and scipy-free, measurably.

The charter's standalone criterion is a claim about what happens with the
rest of the monorepo ABSENT, and the only way to observe that is a process
where it IS absent.  :func:`lxkit.testing.import_isolation` builds one.

Four properties here, and they are four different claims:

* ``import minimax`` drags in NO lorrax package.  That is what makes
  ``services/minimax`` installable on its own, and it is the property the
  extraction bought: the module used to be ``src/common/minimax.py``,
  reached through ``common/__init__.py``, which drags jax in behind its
  back.
* ``import minimax`` needs NO jax.  Unlike ``vcoul``, which imports jax on
  purpose because its q₀ head comes back as jax arrays, this service is
  host numpy end to end.  ``MinimaxNodes`` — the complex128 pytree — stayed
  in ``gw.minimax_screening`` precisely so that this stays true.
* ``import minimax`` needs NO scipy.  scipy is the ``solve`` extra: it is
  reached from the lazy builders behind a PEP-562 ``__getattr__``.  A
  machine with no scipy must still import the package, serve the
  numpy-only ``noncrossing`` rule, and refuse by name.
  ``pyproject.toml`` says so; this is where the claim is measured rather
  than asserted.
* ``import minimax`` does NOT drag in lxkit either.  lxkit is a TEST-time
  dependency here (unlike distrib_la, which depends on it at runtime for
  the capability gates), so it is deliberately NOT handed to the child.

THE ``python -S`` LESSON, measured, not defensive: this repo's venv carries
``site-packages/__editable__.lorrax-0.1.0.pth``, so an ordinary subprocess
of the test interpreter has ``<tree>/src`` on ``sys.path`` no matter what
``PYTHONPATH`` says.  ``-S`` skips ``site`` and therefore every ``.pth``;
the child gets its dependencies back BY NAME, as DIRECTORIES, which carry
no ``.pth`` processing when they arrive through ``PYTHONPATH``.
"""

from __future__ import annotations

import os

import pytest

from lxkit.testing import import_isolation

_TESTS = os.path.dirname(os.path.abspath(__file__))
_SVC_SRC = os.path.join(os.path.dirname(_TESTS), "src")
_SERVICES = os.path.dirname(os.path.dirname(_TESTS))
_REPO = os.path.dirname(_SERVICES)
_LORRAX_SRC = os.path.join(_REPO, "src")

#: minimax's declared RUNTIME dependencies, handed to the child BY NAME.
#: Naming them is the claim, and anything not named — nor in a named
#: package's own declared requirements, which :func:`lxkit.testing.dep_dirs`
#: follows transitively — must stay unreachable.
#:
#: ONE ENTRY, and that is the headline.  ``jax`` is not here (vcoul's is),
#: ``scipy`` is not here (it is the ``solve`` extra), ``lxkit`` is not here
#: (distrib_la's is).  Serving the ``noncrossing`` rule is numpy arithmetic;
#: nothing else should be required to do it, and this tuple is where that is
#: enforced rather than hoped.
_DEPS = ("numpy",)

#: The same, plus the optional solver dependency, for the arms that
#: deliberately exercise the offline half.
_DEPS_WITH_SCIPY = ("numpy", "scipy")

#: What minimax must not touch.  Derived from the monorepo when it is
#: there, so a new top-level lorrax package is covered the day it lands.
_CORE = ("bandstructure", "bse", "centroid", "common", "ffi", "file_io",
         "gw", "isdf", "postprocess", "psp", "runtime", "solvers")

_CPU_PIN = (
    # The child is spawned with a scrubbed path but an INHERITED env; on
    # hosts where jax lives beside no CUDA plugin (Perlmutter's /opt/jax
    # split), an inherited platform request kills the first jax-touching
    # statement.  minimax imports no jax at all, so this pin is belt and
    # braces rather than load-bearing -- but it costs nothing and it means
    # a future accidental jax edge fails on the EDGE and not on device
    # selection, which is the failure that would be misread.
    "import os; os.environ.setdefault('JAX_PLATFORMS', 'cpu')\n")


def _lorrax_roots() -> tuple[str, ...]:
    if os.path.isdir(_LORRAX_SRC):
        found = tuple(sorted(
            n for n in os.listdir(_LORRAX_SRC)
            if os.path.isfile(os.path.join(_LORRAX_SRC, n, "__init__.py"))))
        if found:
            return found
    return _CORE


def _needs_monorepo():
    if not os.path.isdir(_LORRAX_SRC):
        pytest.skip("no lorrax src/ next to this service (standalone "
                    "install); the with-monorepo legs need the checkout")


# ---------------------------------------------------------------------------
# The positive checks
# ---------------------------------------------------------------------------

def test_minimax_imports_with_the_monorepo_absent():
    """The charter criterion, and the tightest one in the phase so far.

    numpy is on the child's path — it is the ONE declared dependency — and
    lorrax is not, in either sense.  ``check_path=True`` asserts both
    halves: ``sys.modules`` proves minimax did not IMPORT lorrax,
    ``sys.path`` proves it COULD not have, and the second is what makes
    the first evidence about the package rather than about this machine.
    """
    run = import_isolation("minimax", _lorrax_roots(), src_dir=_SVC_SRC,
                           deps=_DEPS, check_path=True)
    assert run.file.startswith(_SVC_SRC + os.sep)
    assert run.loaded == () and run.reachable == ()


def test_the_serving_surface_answers_with_no_lorrax_no_jax_and_no_scipy():
    """Not just ``import minimax`` — the whole SERVING path, cold.

    An ``__init__`` that imported cleanly and then failed on first use
    would pass a bare import check and be useless, so the child TOUCHES
    the surface: it counts ``__all__`` and resolves every non-lazy name
    with no scipy arriving, then solves a real ``noncrossing`` request and
    reads its provenance with no jax arriving.  The solve itself may import
    scipy when it is installed: the cache key records scipy's version.
    """
    run = import_isolation(
        "minimax", _lorrax_roots(), src_dir=_SVC_SRC, deps=_DEPS,
        check_path=True,
        preamble=(
            _CPU_PIN +
            "import os, sys\n"
            "os.environ['LORRAX_DISABLE_MINIMAX_DISK_CACHE'] = '1'\n"
            "import minimax as M\n"
            # A PIN, and it is meant to drift only on purpose: the count
            # is what catches an __all__ that quietly emptied, and a
            # deliberate door change is exactly the kind of edit that
            # should have to touch a test.
            "assert len(M.__all__) == 54, (len(M.__all__), M.__all__)\n"
            # Only the NON-lazy half is touched by name here: hasattr on a
            # solver name would fire the PEP-562 __getattr__ and import
            # scipy, which is the very thing the next assertion denies.
            "for _n in M.__all__:\n"
            "    if (_n in M._SOLVER_NAMES or _n in M._FREQUENCY_FIT_NAMES\n"
            "            or _n in M._UNIFORM_RULE_NAMES\n"
            "            or _n in M._LEVELLED_NAMES\n"
            "            or _n in M._ANALYTIC_NAMES\n"
            "            or _n in M._ODD_LAPLACE_NAMES\n"
            "            or _n in M._DAMPED_RULE_NAMES\n"
            "            or _n in M._MATSUBARA_RULE_NAMES\n"
            "            or _n in M._RESPONSE_RULE_NAMES):\n"
            "        continue\n"
            "    assert hasattr(M, _n), _n\n"
            # THE QUARANTINE: the door surface costs no optimiser.
            "assert 'scipy' not in sys.modules, 'the door surface pulled "
            "scipy'\n"
            "q = M.serve(family='noncrossing', target='inverse',\n"
            "            range_value=10.0, error_bound=1e-6, n_max=64)\n"
            "assert q.node_count > 0 and q.nodes.dtype.name == 'float64'\n"
            "assert q.max_error <= 1e-6\n"
            "assert q.provenance.source == 'runtime-uncertified'\n"
            # Measured at the end so it covers everything above.
            "assert 'jax' not in sys.modules, 'minimax pulled jax'\n"))
    assert run.loaded == ()


def test_a_refusal_is_reachable_and_readable_with_no_scipy():
    """A gap must be nameable on a machine that cannot solve anything."""
    run = import_isolation(
        "minimax", _lorrax_roots(), src_dir=_SVC_SRC, deps=_DEPS,
        check_path=True,
        preamble=(
            _CPU_PIN +
            "import sys, minimax as M\n"
            "try:\n"
            "    M.serve(family='complex_laplace', target='complex_laplace',\n"
            "            range_value=21.5, error_bound=1e-6, n_max=64)\n"
            "except M.UncertifiedSolveRefused as e:\n"
            "    t = str(e)\n"
            "    assert 'complex_laplace' in t, t\n"
            "    assert 'no in-process solver' in t, t\n"
            "else:\n"
            "    raise AssertionError('complex_laplace has no solver and "
            "did not refuse')\n"
            "assert 'scipy' not in sys.modules\n"))
    assert run.loaded == ()


def test_the_solver_half_is_reachable_when_scipy_is_there():
    """The lazy door is a DEFERRAL, not a removal.

    ``from minimax import G_hgl`` must work, and it must be the moment
    scipy arrives, not before.  Both halves of that are asserted here, in
    one child, because asserting only the first would be satisfied by an
    eager import.
    """
    pytest.importorskip("scipy")
    run = import_isolation(
        "minimax", _lorrax_roots(), src_dir=_SVC_SRC, deps=_DEPS_WITH_SCIPY,
        check_path=True,
        preamble=(
            _CPU_PIN +
            "import sys, minimax as M\n"
            "assert 'scipy' not in sys.modules, 'import minimax ate scipy'\n"
            "g = M.G_hgl\n"
            "assert 'scipy' in sys.modules, 'the lazy door did not fire'\n"
            "import numpy as np\n"
            "u = np.array([0.5])\n"
            "assert float(g(u)[0]) == float(g(u)[0])\n"
            "assert 'jax' not in sys.modules\n"))
    assert run.loaded == ()


def test_minimax_still_imports_clean_with_lorrax_on_the_path():
    """Isolation must not be an artifact of the monorepo being unreachable:
    with lorrax's src right there, minimax still touches none of it.

    The ``reachable`` assertion is what keeps this cell honest — without it
    a typo in the extra path would make the leg pass by measuring the
    isolated case twice.
    """
    _needs_monorepo()
    run = import_isolation("minimax", _lorrax_roots(), src_dir=_SVC_SRC,
                           deps=_DEPS, extra_path=[_LORRAX_SRC],
                           check_path=False)
    assert run.loaded == ()
    assert {r for r, _ in run.reachable} >= {"gw", "common", "file_io"}, (
        f"lorrax was supposed to be ON the path here; saw {run.reachable}")


# ---------------------------------------------------------------------------
# RED TWINS — the check, shown failing
# ---------------------------------------------------------------------------

def test_the_isolation_check_can_fail():
    """A deliberate lorrax import MUST break the check.

    ``import ffi`` is the probe because ``src/ffi/__init__.py`` is
    stdlib-only at module scope, so the twin fails on the LEAK, not on an
    unrelated ImportError.
    """
    _needs_monorepo()
    with pytest.raises(AssertionError, match="pulled"):
        import_isolation("minimax", _lorrax_roots(), src_dir=_SVC_SRC,
                         deps=_DEPS, extra_path=[_LORRAX_SRC],
                         check_path=False, preamble="import ffi")


def test_the_wrong_copy_of_the_package_is_a_failure():
    """``src_dir`` is the thing under test; resolving the import somewhere
    else (an installed wheel, a stale build dir) measures the wrong file."""
    with pytest.raises(AssertionError, match="not under the src dir"):
        import_isolation("minimax", ("gw",),
                         src_dir=os.path.join(_REPO, "no_such_src"),
                         deps=_DEPS, extra_path=[_SVC_SRC], check_path=False)


def test_the_door_surface_assertion_can_fail():
    """RED TWIN for the surface cell above.

    Asserting a name the door does NOT export must fail inside the child,
    or ``test_the_serving_surface_answers_with_no_lorrax_no_jax_and_no_scipy``
    would pass on any package at all — including one whose ``__all__`` had
    quietly emptied.
    """
    with pytest.raises(AssertionError):
        import_isolation(
            "minimax", _lorrax_roots(), src_dir=_SVC_SRC, deps=_DEPS,
            check_path=True,
            preamble=("import minimax as M\n"
                      "assert hasattr(M, 'solve_laplace_minimax_interval'), "
                      "'the Rydberg rescaler stayed in gw'\n"))


def test_the_scipy_quarantine_assertion_can_fail():
    """RED TWIN for the quarantine.

    ``assert 'scipy' not in sys.modules`` is only evidence if it CAN fail
    in this harness — otherwise a child that never runs the assertion and
    a child that passes it look identical from out here.  Importing scipy
    explicitly, with scipy on the path, must break it.
    """
    pytest.importorskip("scipy")
    with pytest.raises(AssertionError):
        import_isolation(
            "minimax", _lorrax_roots(), src_dir=_SVC_SRC,
            deps=_DEPS_WITH_SCIPY, check_path=True,
            preamble=("import sys, scipy, minimax as M\n"
                      "assert 'scipy' not in sys.modules\n"))

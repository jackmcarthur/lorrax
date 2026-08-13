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
  reached from :mod:`minimax.solver` and nowhere else, behind a PEP-562
  lazy ``__getattr__``.  A machine with no scipy must still import the
  package, serve every certified table in the bundle, and refuse exactly
  the solver names.  ``pyproject.toml`` says so; this is where the claim
  is measured rather than asserted.
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
#: (distrib_la's is).  Serving a certified quadrature is reading a JSON file
#: and an ``.npz``; nothing else should be required to do it, and this tuple
#: is where that is enforced rather than hoped.
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


def test_the_lookup_surface_answers_with_no_lorrax_no_jax_and_no_scipy():
    """Not just ``import minimax`` — the whole SERVING path, cold.

    An ``__init__`` that imported cleanly and then failed on first use
    would pass a bare import check and be useless, so the child TOUCHES
    the surface: it counts ``__all__``, enumerates the catalog, resolves a
    real request against the shipped bundle, reads the payload off disk,
    and prints the provenance line.  Then it asserts that neither jax nor
    scipy arrived while it did any of that — which is the quarantine, and
    it is the reason a production lookup costs microseconds instead of
    importing an optimiser.
    """
    run = import_isolation(
        "minimax", _lorrax_roots(), src_dir=_SVC_SRC, deps=_DEPS,
        check_path=True,
        preamble=(
            _CPU_PIN +
            "import sys\n"
            "import minimax as M\n"
            # A PIN, and it is meant to drift only on purpose: the count
            # is what catches an __all__ that quietly emptied, and a
            # deliberate door change is exactly the kind of edit that
            # should have to touch a test.
            "assert len(M.__all__) == 70, (len(M.__all__), M.__all__)\n"
            # Only the NON-lazy half is touched by name here: hasattr on a
            # solver name would fire the PEP-562 __getattr__ and import
            # scipy, which is the very thing the next assertion denies.
            "for _n in M.__all__:\n"
            "    if _n in M._SOLVER_NAMES or _n in M._FREQUENCY_FIT_NAMES:\n"
            "        continue\n"
            "    assert hasattr(M, _n), _n\n"
            "v = M.catalog()\n"
            "assert len(v) == 31, len(v)\n"
            "assert v.families() == ('crossing', 'noncrossing'), v\n"
            "q = M.lookup(family='noncrossing', target='inverse',\n"
            "             range_value=10.0, error_bound=1e-6, n_max=64)\n"
            "assert q.node_count == 7 and q.nodes.dtype.name == 'float64'\n"
            "assert q.provenance.source == 'shipped'\n"
            "assert q.provenance.table_hash.startswith('sha256:')\n"
            "assert 'shipped' in q.one_line()\n"
            # THE QUARANTINE, measured at the end so it covers everything
            # above rather than only the import.
            "assert 'jax' not in sys.modules, 'minimax pulled jax'\n"
            "assert 'scipy' not in sys.modules, 'a lookup pulled scipy'\n"))
    assert run.loaded == ()


def test_a_refusal_is_reachable_and_readable_with_no_scipy():
    """A gap must be nameable on a machine that cannot solve anything.

    This is R1's shape as a property of the install: the refusal path is
    the one that must work when the solver is absent, because "no table
    and no solver" is exactly the situation where a silent fallback used
    to be most expensive.  The message must carry the nearest certified
    artifact and BOTH levers.
    """
    run = import_isolation(
        "minimax", _lorrax_roots(), src_dir=_SVC_SRC, deps=_DEPS,
        check_path=True,
        preamble=(
            _CPU_PIN +
            "import sys, minimax as M\n"
            "try:\n"
            "    M.lookup(family='crossing', target='hgl', range_value=83.0,\n"
            "             error_bound=1e-6, n_max=500, eps_q=1e-3)\n"
            "except M.NoCertifiedTable as e:\n"
            "    t = str(e)\n"
            "    assert 'A_dim=83' in t, t\n"
            "    assert 'nearest certified below' in t, t\n"
            "    assert 'reachable by' in t, t\n"
            "    assert 'or generate' in t, t\n"
            "else:\n"
            "    raise AssertionError('A_dim=83 is outside the catalog and "
            "did not refuse')\n"
            "assert 'scipy' not in sys.modules\n"))
    assert run.loaded == ()


def test_the_solver_half_is_reachable_when_scipy_is_there():
    """The lazy door is a DEFERRAL, not a removal.

    ``from minimax import G_hgl`` must work — the generator campaign and
    the certification tier need those names — and it must be the moment
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
    or ``test_the_lookup_surface_answers_with_no_lorrax_no_jax_and_no_scipy``
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

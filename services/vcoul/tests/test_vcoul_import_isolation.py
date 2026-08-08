"""Standalone, falsifiably.

The charter's standalone criterion is a claim about what happens with the
rest of the monorepo ABSENT, and the only way to observe that is a process
where it IS absent.  :func:`lxkit.testing.import_isolation` builds one.

Three properties here, and they are three different claims:

* ``import vcoul`` drags in NO lorrax package.  That is what makes
  ``services/vcoul`` installable on its own, and it is the property the
  extraction actually bought: this code used to say ``from common import
  Meta`` in four modules and reach into a ``wfn`` loader in five more.
* ``import vcoul`` needs NO scipy.  scipy is the service's one optional
  dependency, quarantined in :mod:`vcoul.minibz` behind the
  announce-or-refuse gate; a machine without it must still import the
  package, evaluate every ``v(q+G)``, and refuse exactly one generator BY
  NAME.  ``pyproject.toml`` says so in the ``sobol`` extra, and this is
  where that claim is measured rather than asserted.
* ``import vcoul`` does NOT drag in lxkit either.  lxkit is a TEST-time
  dependency here (unlike distrib_la, which depends on it at runtime for
  the capability gates), so it is deliberately NOT handed to the child.

Unlike lxkit, vcoul DOES import jax, and that is correct rather than a
concession: ``wrap_points_to_voronoi`` is a jitted kernel and the
``q0_average`` pair comes back as jax arrays because the head resolver
consumes them inside a jax computation.  So jax is handed to the child on
purpose.

THE ``python -S`` LESSON, measured, not defensive: this repo's venv carries
``site-packages/__editable__.lorrax-0.1.0.pth``, so an ordinary subprocess
of the test interpreter has ``<tree>/src`` on ``sys.path`` no matter what
``PYTHONPATH`` says.  ``-S`` skips ``site`` and therefore every ``.pth``;
the child gets its dependencies back BY NAME, as DIRECTORIES, which carry
no ``.pth`` processing when they arrive through ``PYTHONPATH``.
:func:`import_isolation` does all of that; these cells choose the NAMES.
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

#: vcoul's declared runtime dependencies, handed to the child BY NAME
#: through ``deps=``.  Naming them is the claim, and anything not named —
#: nor in a named package's own declared requirements, which
#: :func:`lxkit.testing.dep_dirs` follows transitively — must stay
#: unreachable.  numpy arrives through jax's own requirements.
#:
#: ``lxkit`` is NOT here, and that is the difference from distrib_la:
#: vcoul owns no ``.so``, probes no device and counts no processes, so it
#: has no capability gate to share and no runtime need of the foundation.
_DEPS = ("jax",)

#: What vcoul must not touch.  Derived from the monorepo when it is
#: there, so a new top-level lorrax package is covered the day it lands.
_CORE = ("bandstructure", "bse", "centroid", "common", "ffi", "file_io",
         "gw", "isdf", "postprocess", "psp", "runtime", "solvers")


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


def _needs_jax():
    pytest.importorskip("jax")


# ---------------------------------------------------------------------------
# The positive checks
# ---------------------------------------------------------------------------

def test_vcoul_imports_with_the_monorepo_absent():
    """The charter criterion.  jax is on the child's path — it is a
    declared dependency — and lorrax is not, in either sense."""
    _needs_jax()
    # check_path=True: BOTH halves.  sys.modules proves vcoul did not
    # IMPORT lorrax; sys.path proves it COULD not have, and the second is
    # what makes the first evidence about the package rather than about
    # this machine.  It holds even with site-packages handed back, because
    # ``python -S`` skips the editable-install .pth that would otherwise
    # put <tree>/src on every subprocess's path.
    run = import_isolation("vcoul", _lorrax_roots(), src_dir=_SVC_SRC,
                           deps=_DEPS, check_path=True)
    assert run.file.startswith(_SVC_SRC + os.sep)
    assert run.loaded == () and run.reachable == ()


def test_the_whole_public_surface_answers_with_no_lorrax():
    """Not just ``import vcoul`` — every name on the door.

    An ``__init__`` that imported cleanly and then failed on first use
    would pass a bare import check and be useless, so the child TOUCHES
    the surface: it counts ``__all__``, evaluates a v(q+G) table through
    the driver, and builds a geometry from a duck-typed stand-in for the
    ``wfn`` loader this package is no longer allowed to know about.
    """
    _needs_jax()
    run = import_isolation(
        "vcoul", _lorrax_roots(), src_dir=_SVC_SRC, deps=_DEPS,
        check_path=True,
        preamble=(
            "import numpy as np\n"
            "import vcoul as V\n"
            "assert len(V.__all__) == 30, V.__all__\n"
            "for _n in V.__all__:\n"
            "    assert hasattr(V, _n), _n\n"
            "class W:\n"
            "    blat = 2.0\n"
            "    bvec = np.eye(3)\n"
            "    cell_volume = 8.0\n"
            "g = V.CoulombGeometry.from_wfn(W())\n"
            "assert abs(g.bvec[0, 0] - 2.0) < 1e-15\n"
            "t = V.v_qG_table(V.get_kernel(3), np.array([[0.25, 0.0, 0.0]]),\n"
            "                 np.zeros((1, 3, 4)), geometry=g)\n"
            "assert t.shape == (1, 4) and t.dtype == np.float64\n"
            "assert float(t[0, 0]) > 0.0\n"
            # The head path too, standalone: the mini-BZ draw and the
            # argmin injection are the service's one physics-moving
            # surface, and a lorrax-free child is where "pure math" is
            # measured rather than claimed.
            "h = V.build_v_head_miniBZ_fn_3d((2, 2, 2), g.bvec,\n"
            "                                g.cell_volume, nmc=32)\n"
            "K = np.array([[0.5, 0.0, 0.0]])\n"
            "assert float(h(K)[0]) > 0.0\n"
            "assert float(h(K)[0]) == float(h(-K)[0])\n"
            "t2 = V.v_qG_table(V.get_kernel(3),\n"
            "                  np.array([[0.25, 0.0, 0.0]]),\n"
            "                  np.zeros((1, 3, 4)), geometry=g,\n"
            "                  v_head_fn=h)\n"
            "assert float(t2[0, 0]) != float(t[0, 0])\n"))
    assert run.loaded == ()


def test_vcoul_imports_and_computes_with_no_scipy():
    """scipy is OPTIONAL, and the uniform arm proves it.

    The child has jax and no scipy at all.  ``method='uniform'`` must
    still draw, and ``method='sobol'`` must REFUSE by name rather than
    quietly serving a different generator — which is the whole point of
    the announce-or-refuse gate and the one behaviour the extraction
    deliberately changed.
    """
    _needs_jax()
    run = import_isolation(
        "vcoul", _lorrax_roots(), src_dir=_SVC_SRC, deps=_DEPS,
        check_path=True,
        preamble=(
            "import sys\n"
            "assert 'scipy' not in sys.modules\n"
            "import numpy as np, vcoul as V\n"
            "b = 2.0 * np.pi * np.eye(3)\n"
            "u = V.minibz_voronoi_batches(b, (2, 2, 2), nsamples=64,\n"
            "                             method='uniform')\n"
            "assert len(u) == 1 and u[0].shape == (64, 3)\n"
            "try:\n"
            "    import scipy.stats.qmc  # noqa: F401\n"
            "except Exception:\n"
            "    try:\n"
            "        V.minibz_voronoi_batches(b, (2, 2, 2), nsamples=64,\n"
            "                                 method='sobol')\n"
            "    except RuntimeError as e:\n"
            "        assert 'REFUSAL' in str(e) and 'install scipy' in str(e)\n"
            "    else:\n"
            "        raise AssertionError('sobol did not refuse without scipy')\n"))
    assert run.loaded == ()


def test_vcoul_still_imports_clean_with_lorrax_on_the_path():
    """Isolation must not be an artifact of the monorepo being unreachable:
    with lorrax's src right there, vcoul still touches none of it.

    The ``reachable`` assertion is what keeps this cell honest — without it
    a typo in the extra path would make the leg pass by measuring the
    isolated case twice.
    """
    _needs_monorepo()
    _needs_jax()
    run = import_isolation("vcoul", _lorrax_roots(), src_dir=_SVC_SRC,
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
    _needs_jax()
    with pytest.raises(AssertionError, match="pulled"):
        import_isolation("vcoul", _lorrax_roots(), src_dir=_SVC_SRC,
                         deps=_DEPS, extra_path=[_LORRAX_SRC],
                         check_path=False, preamble="import ffi")


def test_the_wrong_copy_of_the_package_is_a_failure():
    """``src_dir`` is the thing under test; resolving the import somewhere
    else (an installed wheel, a stale build dir) measures the wrong file."""
    _needs_jax()
    with pytest.raises(AssertionError, match="not under the src dir"):
        import_isolation("vcoul", ("gw",),
                         src_dir=os.path.join(_REPO, "no_such_src"),
                         deps=_DEPS, extra_path=[_SVC_SRC], check_path=False)


def test_the_door_surface_assertion_can_fail():
    """RED TWIN for the surface cell above.

    Asserting a name the door does NOT export must fail inside the child,
    or ``test_the_whole_public_surface_answers_with_no_lorrax`` would pass
    on any package at all — including one whose ``__all__`` had quietly
    emptied.
    """
    _needs_jax()
    with pytest.raises(AssertionError):
        import_isolation(
            "vcoul", _lorrax_roots(), src_dir=_SVC_SRC, deps=_DEPS,
            check_path=True,
            preamble=("import vcoul as V\n"
                      "assert hasattr(V, 'compute_all_V_q'), "
                      "'the dispatcher stayed in gw'\n"))

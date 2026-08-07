"""Standalone, falsifiably.

Two properties, one mechanism (:func:`lxkit.testing.import_isolation`):

* ``import lxkit`` drags in no LORRAX package — the charter's standalone
  criterion, and the thing that makes ``services/lxkit`` installable on its
  own;
* ``import lxkit`` drags in no JAX — the PINNED PROPERTY of this package
  (see its docstring): ``Gate.enabled()`` is a kernel-cache key read before
  ``jax.distributed.initialize``, so a jax import at lxkit import time would
  initialize the backend too early and destroy that promise.

The child runs under ``python -S`` and gets exactly the path this file hands
it.  That is not paranoia: this repo's venv carries
``site-packages/__editable__.lorrax-0.1.0.pth``, so an ordinary subprocess of
the test interpreter has ``<tree>/src`` on ``sys.path`` no matter what
``PYTHONPATH`` says — the hazard survey 3 (c4) names, measured here.  Legs
that need pytest or jax ask for them BY NAME through ``deps=``; a directory
carries no ``.pth`` processing when it arrives through ``PYTHONPATH``.

NAMING them is not a spelling preference.  These three cells used to pass
``sysconfig.get_paths()["purelib"]``, which is the obvious answer and is
WRONG inside the Perlmutter Shifter image, where jax is a source checkout at
``/opt/jax`` reached through an editable finder hook that ``python -S``
skips.  The child then had no jax, died before printing its payload line,
and three cells with nothing to do with isolation went red on the only
machine that runs the FFI stack.  ``lxkit.testing.dep_dirs`` asks each
dependency where it actually lives; see its docstring for the measurement.

RED TWINS ARE MANDATORY (charter, falsification doctrine).  Both halves of
the check — the ``sys.modules`` half and the ``sys.path`` half — are shown
here to FAIL when they should.  A check that cannot fail is a tautology, and
this file would be its own best example without them.
"""

from __future__ import annotations

import os

import pytest

from lxkit.testing import import_isolation

_TESTS = os.path.dirname(os.path.abspath(__file__))
_LXKIT_SRC = os.path.join(os.path.dirname(_TESTS), "src")
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_TESTS)))
_LORRAX_SRC = os.path.join(_REPO, "src")
#: What a child needs back to import the module under test at all.  Handed
#: over BY NAME (see the module docstring); anything not named here — and
#: not in a named package's own DECLARED requirements, which
#: :func:`lxkit.testing.dep_dirs` follows — must be unreachable in the
#: child, which is what the isolation check measures.
#:
#: ONE NAME EACH, on purpose.  A hand-kept transitive list is a losing game
#: with a quiet failure mode: this pair used to spell out pluggy/iniconfig/
#: packaging and STILL died inside the Perlmutter image on `import
#: pygments`, three frames deep in pytest's terminal writer.  The installer
#: wrote the closure down; dep_dirs reads it.
_PYTEST_DEPS = ("pytest",)
_JAX_DEPS = ("jax",)

#: Roots a standalone lxkit must not touch.  Derived from the monorepo when
#: it is there (so a new top-level package is covered the day it lands),
#: falling back to the named core for a bare standalone install.
_CORE = ("bandstructure", "bse", "centroid", "common", "ffi", "file_io",
         "gw", "isdf", "runtime")


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

def test_lxkit_imports_with_the_monorepo_absent():
    run = import_isolation("lxkit", _lorrax_roots(), src_dir=_LXKIT_SRC)
    assert run.file.startswith(_LXKIT_SRC + os.sep)
    assert run.loaded == () and run.reachable == ()


def test_the_pytest_plugin_imports_with_the_monorepo_absent():
    """``lxkit.testing`` is loaded by pytest at startup in every suite that
    installs lxkit, so it is the module most able to poison a session.  It
    needs pytest, and nothing else."""
    run = import_isolation("lxkit.testing", _lorrax_roots(),
                           src_dir=_LXKIT_SRC, deps=_PYTEST_DEPS)
    assert run.loaded == () and run.reachable == ()


def test_importing_lxkit_pulls_in_no_jax():
    """THE pinned property.  jax is handed to the child on purpose and
    ``check_path=False``: the claim is not that jax is missing, it is that
    lxkit does not reach for it."""
    forbidden = ("jax", "jaxlib", "numpy", "scipy", "h5py")
    run = import_isolation("lxkit", forbidden, src_dir=_LXKIT_SRC,
                           deps=_JAX_DEPS, check_path=False)
    assert run.loaded == ()
    if not any(root == "jax" for root, _ in run.reachable):
        pytest.skip("jax is not installed in this interpreter, so 'lxkit "
                    "did not import it' is not evidence of anything")


def test_the_pytest_plugin_pulls_in_no_jax_either():
    """The plugin loads in EVERY pytest session of a suite that installs
    lxkit, including the WSL shape-algebra legs that have no jax at all."""
    run = import_isolation("lxkit.testing", ("jax", "jaxlib"),
                           src_dir=_LXKIT_SRC,
                           deps=_PYTEST_DEPS + _JAX_DEPS, check_path=False)
    assert run.loaded == ()
    if not any(root == "jax" for root, _ in run.reachable):
        pytest.skip("no jax installed; nothing could have been pulled in")


def test_lxkit_still_imports_clean_with_the_whole_monorepo_on_the_path():
    """Isolation must not be an artifact of the monorepo being unreachable:
    with lorrax's src right there, lxkit still touches none of it.

    The ``reachable`` assertion is what keeps this cell honest — without it
    a typo in the extra path would make the leg pass by measuring the
    isolated case twice."""
    _needs_monorepo()
    run = import_isolation("lxkit", _lorrax_roots(), src_dir=_LXKIT_SRC,
                           extra_path=[_LORRAX_SRC], check_path=False)
    assert run.loaded == ()
    assert {r for r, _ in run.reachable} >= {"gw", "ffi", "common"}, (
        f"lorrax was supposed to be ON the path here; saw {run.reachable}")


# ---------------------------------------------------------------------------
# RED TWINS — each half of the check, shown failing
# ---------------------------------------------------------------------------

def test_the_sys_modules_half_can_fail():
    """A deliberate forbidden import MUST break the check.

    ``import ffi`` is the probe because ``src/ffi/__init__.py`` at 96a6399
    is stdlib-only at module scope (its imports are inside ``ffi_dial_key``),
    so the twin fails on the LEAK, not on an unrelated ImportError."""
    _needs_monorepo()
    with pytest.raises(AssertionError, match="pulled"):
        import_isolation("lxkit", _lorrax_roots(), src_dir=_LXKIT_SRC,
                         extra_path=[_LORRAX_SRC], check_path=False,
                         preamble="import ffi")


def test_the_sys_path_half_can_fail():
    """With the monorepo reachable, a clean ``sys.modules`` proves nothing —
    so the strict check refuses to call that run isolated."""
    _needs_monorepo()
    with pytest.raises(AssertionError, match="importable from sys.path"):
        import_isolation("lxkit", _lorrax_roots(), src_dir=_LXKIT_SRC,
                         extra_path=[_LORRAX_SRC], check_path=True)


def test_the_jax_check_can_fail():
    """Same twin for the pinned property: an import of jax in the preamble
    must be caught."""
    pytest.importorskip("jax")
    with pytest.raises(AssertionError, match="pulled"):
        import_isolation("lxkit", ("jax",), src_dir=_LXKIT_SRC,
                         deps=_JAX_DEPS, check_path=False,
                         preamble="import jax")


def test_a_probe_that_never_prints_is_a_failure_not_a_pass():
    """Judge by artifacts, not exit codes.  A preamble that kills the child
    before the payload line must not read as green."""
    with pytest.raises(AssertionError, match="produced no LXKIT_ISOLATION"):
        import_isolation("lxkit", ("jax",), src_dir=_LXKIT_SRC,
                         check_path=False,
                         preamble="import sys; sys.exit(0)")


def test_the_wrong_copy_of_the_package_is_a_failure():
    """``src_dir`` is the thing under test; resolving the import somewhere
    else (an installed wheel, a stale build dir) measures the wrong file."""
    with pytest.raises(AssertionError, match="not under the src dir"):
        import_isolation("lxkit", ("jax",),
                         src_dir=os.path.join(_REPO, "no_such_src"),
                         extra_path=[_LXKIT_SRC], check_path=False)


def test_an_empty_forbidden_list_is_refused_outright():
    with pytest.raises(ValueError, match="cannot fail"):
        import_isolation("lxkit", (), src_dir=_LXKIT_SRC)

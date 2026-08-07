"""Standalone, falsifiably.

The charter's standalone criterion is a claim about what happens with the
rest of the monorepo ABSENT, and the only way to observe that is a process
where it IS absent.  :func:`lxkit.testing.import_isolation` builds one.

Two properties here, and they are different claims:

* ``import distrib_la`` drags in NO lorrax package.  That is what makes
  ``services/distrib_la`` installable on its own and what "only distrib_la
  sees scalapack/slate/cusolvermp" is worth.
* ``import distrib_la`` needs NO ``.so``.  A deck parser reads
  ``BACKEND_CHOICES`` and must not need the FFI layer to do it
  (``gw.gw_config.eigh_backend_choices`` is exactly that caller), and the
  shape-algebra test tier has to run on a laptop.

Unlike lxkit, distrib_la DOES import jax, and that is correct rather than a
concession: every public entry point takes a ``jax.sharding.Mesh``.  So jax
is handed to the child on purpose.

THE ``python -S`` LESSON, measured, not defensive: this repo's venv carries
``site-packages/__editable__.lorrax-0.1.0.pth``, so an ordinary subprocess
of the test interpreter has ``<tree>/src`` on ``sys.path`` no matter what
``PYTHONPATH`` says.  ``-S`` skips ``site`` and therefore every ``.pth``;
the child gets its dependencies back BY NAME, as DIRECTORIES, which carry
no ``.pth`` processing when they arrive through ``PYTHONPATH``.
:func:`import_isolation` does all of that; these cells choose the NAMES --
and where a name actually resolves to is the second measured lesson, which
now lives with the rest of the harness policy in
:func:`lxkit.testing.dep_dirs` (it is not ``purelib`` inside the Perlmutter
Shifter image, and assuming it was cost three lxkit cells a red).
"""

from __future__ import annotations

import os

import pytest

from lxkit.testing import import_isolation

_TESTS = os.path.dirname(os.path.abspath(__file__))
_SVC_SRC = os.path.join(os.path.dirname(_TESTS), "src")
_SERVICES = os.path.dirname(os.path.dirname(_TESTS))
_LXKIT_SRC = os.path.join(_SERVICES, "lxkit", "src")
_REPO = os.path.dirname(_SERVICES)
_LORRAX_SRC = os.path.join(_REPO, "src")

#: distrib_la's declared runtime dependencies, handed to the child BY NAME
#: through ``deps=``.  Naming them is the claim, and anything not named —
#: nor in a named package's own declared requirements, which
#: :func:`lxkit.testing.dep_dirs` follows transitively — must stay
#: unreachable.  dep_dirs resolves each to where it ACTUALLY lives, which
#: is not ``purelib`` inside the Perlmutter Shifter image (jax there is a
#: source checkout at /opt/jax behind an editable finder hook that
#: ``python -S`` skips); that measurement lives in lxkit with the rest of
#: the harness policy.
_DEPS = ("jax",)

#: What distrib_la must not touch.  Derived from the monorepo when it is
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

def test_distrib_la_imports_with_the_monorepo_absent():
    """The charter criterion.  lxkit and jax are on the child's path — they
    are declared dependencies — and lorrax is not, in either sense."""
    _needs_jax()
    # check_path=True: BOTH halves.  sys.modules proves distrib_la did not
    # IMPORT lorrax; sys.path proves it COULD not have, and the second is
    # what makes the first evidence about the package rather than about
    # this machine.  It holds even with site-packages handed back, because
    # ``python -S`` skips the editable-install .pth that would otherwise
    # put <tree>/src on every subprocess's path.
    run = import_isolation("distrib_la", _lorrax_roots(), src_dir=_SVC_SRC,
                           deps=_DEPS, extra_path=[_LXKIT_SRC], check_path=True)
    assert run.file.startswith(_SVC_SRC + os.sep)
    assert run.loaded == () and run.reachable == ()


def test_the_backend_vocabulary_reads_with_no_so_anywhere():
    """``BACKEND_CHOICES`` is importable with no FFI library on the machine.

    ``gw.gw_config.eigh_backend_choices`` reads it at DECK-PARSE time and
    falls back to a literal tuple "if ffi cannot be imported", so the
    vocabulary has to be answerable before any ``.so`` question is asked.
    Pinned by pointing the loader's two env vars at paths that do not
    exist: those are REFUSALS in :func:`distrib_la.loader._locate_so` (an
    explicit pin that cannot be honored never falls through), so if merely
    naming the vocabulary touched the loader this cell would fail.
    """
    _needs_jax()
    preamble = (
        "import os\n"
        "os.environ['LORRAX_FFI_SO'] = '/nonexistent/liblorrax_ffi.so'\n"
        "os.environ['LORRAX_FFI_HOST_SO'] = "
        "'/nonexistent/liblorrax_ffi_host.so'\n")
    run = import_isolation(
        "distrib_la", _lorrax_roots(), src_dir=_SVC_SRC,
        deps=_DEPS, extra_path=[_LXKIT_SRC], check_path=True, preamble=preamble
        + "import distrib_la as _d\n"
          "assert _d.BACKEND_CHOICES['eigh'][0] == 'auto'\n"
          "assert 'native2d' in _d.BACKEND_CHOICES['cholesky']\n"
          "assert _d.OPS == ('eigh', 'cholesky', 'solve_lu')\n")
    assert run.loaded == ()


def test_list_backends_never_raises_without_a_library():
    """The never-raising report, with no ``.so`` and a broken pin.

    ``runtime.initialize_communicator_stack``'s startup banner calls it on
    every run; a report that can take the run down is worse than no report.
    Each row must still carry the three-way probe REASON, or the banner
    says "unavailable" and nothing actionable.
    """
    import jax
    import numpy as np
    from jax.sharding import Mesh

    import distrib_la as D

    # RESTORE, never delete.  This cell used to `del os.environ[...]` in
    # its finally block, which is not the inverse of setting it: on a
    # machine where the variable WAS set -- i.e. Perlmutter, under the
    # BUILD_NOTES pins -- it left every later cell in the process with no
    # host .so pin at all.  MEASURED 2026-08-07: under a single-worker
    # `-m distrib_la` run the five test_so_acceptance cells that follow
    # went from checking a real library to
    #
    #     SKIPPED: LORRAX_FFI_HOST_SO is not set
    #
    # and the C++ acceptance tier evaporated silently.  The skip-honesty
    # gate is what found it, which is the entire argument for that gate.
    _saved = os.environ.get("LORRAX_FFI_HOST_SO")
    os.environ["LORRAX_FFI_HOST_SO"] = "/nonexistent/liblorrax_ffi_host.so"
    try:
        mesh = Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1),
                    ("x", "y"))
        for op in D.OPS:
            status = D.list_backends(op, mesh)
            assert status["native"].startswith("available")
            for backend, text in status.items():
                if text.startswith("available"):
                    continue
                assert "unavailable:" in text and len(text) > 40, (
                    f"{op}/{backend} refuses without saying why: {text!r}")
    finally:
        if _saved is None:
            os.environ.pop("LORRAX_FFI_HOST_SO", None)
        else:
            os.environ["LORRAX_FFI_HOST_SO"] = _saved


def test_distrib_la_still_imports_clean_with_lorrax_on_the_path():
    """Isolation must not be an artifact of the monorepo being unreachable:
    with lorrax's src right there, distrib_la still touches none of it.

    The ``reachable`` assertion is what keeps this cell honest — without it
    a typo in the extra path would make the leg pass by measuring the
    isolated case twice.
    """
    _needs_monorepo()
    _needs_jax()
    run = import_isolation("distrib_la", _lorrax_roots(), src_dir=_SVC_SRC,
                           deps=_DEPS, extra_path=[_LXKIT_SRC, _LORRAX_SRC], check_path=False)
    assert run.loaded == ()
    assert {r for r, _ in run.reachable} >= {"gw", "ffi", "common"}, (
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
        import_isolation("distrib_la", _lorrax_roots(), src_dir=_SVC_SRC,
                         deps=_DEPS, extra_path=[_LXKIT_SRC, _LORRAX_SRC],
                         check_path=False, preamble="import ffi")


def test_the_wrong_copy_of_the_package_is_a_failure():
    """``src_dir`` is the thing under test; resolving the import somewhere
    else (an installed wheel, a stale build dir) measures the wrong file."""
    _needs_jax()
    with pytest.raises(AssertionError, match="not under the src dir"):
        import_isolation("distrib_la", ("gw",),
                         src_dir=os.path.join(_REPO, "no_such_src"),
                         deps=_DEPS, extra_path=[_SVC_SRC, _LXKIT_SRC], check_path=False)


def test_the_loader_pins_survive_this_file():
    """RED TWIN for the restore above, and for every cell here that
    touches the environment.

    A test that leaves ``LORRAX_FFI_HOST_SO`` unset poisons everything
    after it in the process: the C++ acceptance tier stops finding a
    library to check and reports five clean-looking skips.  That is not
    hypothetical — it is what this file did until 2026-08-07, caught by
    the skip-honesty gate on a single-worker Perlmutter run.

    Asserting the value is UNCHANGED rather than merely present is the
    part that matters: restoring the wrong path would be the same defect
    wearing a plausible value.
    """
    before = dict(os.environ)
    test_list_backends_never_raises_without_a_library()
    for var in ("LORRAX_FFI_SO", "LORRAX_FFI_HOST_SO"):
        assert os.environ.get(var) == before.get(var), (
            f"{var} changed from {before.get(var)!r} to "
            f"{os.environ.get(var)!r}; a cell that edits a loader pin must "
            f"put it back exactly, or every later cell measures a "
            f"different machine")

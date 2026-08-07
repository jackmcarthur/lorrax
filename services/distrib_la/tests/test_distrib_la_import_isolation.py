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
:func:`import_isolation` does all of that; these cells choose the paths --
and choosing them is where the second measured lesson lives, in
:func:`_dep_dirs`.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from lxkit.testing import import_isolation

_TESTS = os.path.dirname(os.path.abspath(__file__))
_SVC_SRC = os.path.join(os.path.dirname(_TESTS), "src")
_SERVICES = os.path.dirname(os.path.dirname(_TESTS))
_LXKIT_SRC = os.path.join(_SERVICES, "lxkit", "src")
_REPO = os.path.dirname(_SERVICES)
_LORRAX_SRC = os.path.join(_REPO, "src")

#: distrib_la's declared runtime dependency, plus what jax itself needs to
#: import.  The child gets these back BY NAME, which is the whole point:
#: naming them is the claim, and anything not named must be unreachable.
_DEPS = ("jax", "jaxlib", "numpy", "scipy", "ml_dtypes", "opt_einsum")


def _dep_dirs() -> list[str]:
    """The directories THIS interpreter resolved :data:`_DEPS` from.

    NOT ``sysconfig.get_paths()['purelib']``, which is the obvious answer
    and is WRONG on the machine that matters.  Measured inside the
    Perlmutter Shifter image (``lorrax_J070``, 2026-08-07): ``purelib`` is
    ``/usr/local/lib/python3.12/dist-packages`` and ``jax.__file__`` is
    ``/opt/jax/jax/__init__.py`` — the image's jax is a source checkout
    reached through an editable finder hook, so a child handed ``purelib``
    and run under ``python -S`` has NO jax, ``import distrib_la`` dies
    before printing, and the isolation cells fail for a reason that has
    nothing to do with isolation.

    Asking each dependency where it actually lives is right in both places
    (a venv answers ``purelib`` for all of them) and it keeps the child's
    path a list of names rather than a guess about layout.
    """
    out = []
    for name in _DEPS:
        try:
            spec = importlib.util.find_spec(name)
        except Exception:                                      # noqa: BLE001
            continue
        if spec is None or not spec.origin or spec.origin == "namespace":
            continue
        pkg = os.path.dirname(os.path.realpath(spec.origin))
        root = os.path.dirname(pkg) if os.path.basename(pkg) == name else pkg
        if root not in out:
            out.append(root)
    return out


#: What the child needs to import distrib_la at all: its lxkit foundation
#: and jax.  Everything else stays off the path — that is the measurement.
DEPS = [_LXKIT_SRC] + _dep_dirs()

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
                           extra_path=DEPS, check_path=True)
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
        extra_path=DEPS, check_path=True, preamble=preamble
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
        del os.environ["LORRAX_FFI_HOST_SO"]


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
                           extra_path=DEPS + [_LORRAX_SRC], check_path=False)
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
                         extra_path=DEPS + [_LORRAX_SRC],
                         check_path=False, preamble="import ffi")


def test_the_wrong_copy_of_the_package_is_a_failure():
    """``src_dir`` is the thing under test; resolving the import somewhere
    else (an installed wheel, a stale build dir) measures the wrong file."""
    _needs_jax()
    with pytest.raises(AssertionError, match="not under the src dir"):
        import_isolation("distrib_la", ("gw",),
                         src_dir=os.path.join(_REPO, "no_such_src"),
                         extra_path=[_SVC_SRC] + DEPS, check_path=False)

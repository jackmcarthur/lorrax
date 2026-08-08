"""Standalone, falsifiably.

The charter's standalone criterion is a claim about what happens with the
rest of the monorepo ABSENT, and the only way to observe that is a process
where it IS absent.  :func:`lxkit.testing.import_isolation` builds one.

Two properties here, and they are different claims:

* ``import wfn_loader`` drags in NO lorrax package.  That is what makes
  ``services/wfn_loader`` installable on its own, and it is the property
  the extraction had to buy with three edits and no others: the module-top
  ``common.shard_map`` / ``common.collectives`` imports became the private
  copies in this package, and ``common.gvec_fft_box`` / ``file_io.mf_header``
  / ``file_io.slab_io`` became LAZY, inside the one method each is
  used by.
* ``import wfn_loader`` needs no ``.so`` and no MPI.  The collective phdf5
  read is reached through the slab_io door at CALL time; a metadata-only
  caller (``psp.get_DFT_mtxels`` builds a loader just to read ``gvecs``)
  must not need the FFI layer to open a file.  Nothing here names phdf5 at
  module scope, so this property is a corollary of the first and the cells
  below measure it as one.

Unlike lxkit, wfn_loader DOES import jax, numpy and h5py, and that is
correct rather than a concession: ``load`` returns a ``jax.Array`` and the
serial read is its own ``h5py`` hyperslab.  So all three are handed to the
child on purpose, BY NAME.

THE ``python -S`` LESSON, measured, not defensive: this repo's venv carries
``site-packages/__editable__.lorrax-0.1.0.pth``, so an ordinary subprocess
of the test interpreter has ``<tree>/src`` on ``sys.path`` no matter what
``PYTHONPATH`` says.  ``-S`` skips ``site`` and therefore every ``.pth``;
the child gets its dependencies back BY NAME, as DIRECTORIES, which carry
no ``.pth`` processing when they arrive through ``PYTHONPATH``.
:func:`import_isolation` does all of that; these cells choose the NAMES --
and where a name actually resolves to is the second measured lesson, which
lives with the rest of the harness policy in
:func:`lxkit.testing.dep_dirs`.
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

#: wfn_loader's declared runtime dependencies, handed to the child BY NAME
#: through ``deps=``.  Naming them is the claim, and anything not named —
#: nor in a named package's own declared requirements, which
#: :func:`lxkit.testing.dep_dirs` follows transitively — must stay
#: unreachable.  These three are exactly ``pyproject.toml``'s
#: ``dependencies`` minus ``lxkit``, which arrives as a source directory
#: through ``extra_path`` because it is in-tree here.
_DEPS = ("jax", "numpy", "h5py")

#: What wfn_loader must not touch.  Derived from the monorepo when it is
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


def _needs_deps():
    for name in _DEPS:
        pytest.importorskip(name)


# ---------------------------------------------------------------------------
# The positive checks
# ---------------------------------------------------------------------------

def test_wfn_loader_imports_with_the_monorepo_absent():
    """The charter criterion.  lxkit, jax, numpy and h5py are on the child's
    path — they are declared dependencies — and lorrax is not, in either
    sense."""
    _needs_deps()
    # check_path=True: BOTH halves.  sys.modules proves wfn_loader did not
    # IMPORT lorrax; sys.path proves it COULD not have, and the second is
    # what makes the first evidence about the package rather than about
    # this machine.  It holds even with site-packages handed back, because
    # ``python -S`` skips the editable-install .pth that would otherwise
    # put <tree>/src on every subprocess's path.
    run = import_isolation("wfn_loader", _lorrax_roots(), src_dir=_SVC_SRC,
                           deps=_DEPS, extra_path=[_LXKIT_SRC], check_path=True)
    assert run.file.startswith(_SVC_SRC + os.sep)
    assert run.loaded == () and run.reachable == ()


def test_every_door_name_binds_with_no_lorrax_anywhere():
    """The door's whole export list, resolved in the isolated process.

    ``import wfn_loader`` succeeding is weaker than it looks: a door that
    re-exported nothing would pass the cell above.  This one names every
    entry of ``__all__``, which is the SAME list the transitional shim
    ``src/file_io/wfn_loader.py`` re-imports — so if an extraction ever
    drops a helper from the door, the failure lands here, in the service's
    own suite, rather than as an ImportError in lorrax's package
    ``__init__``.

    The underscored names are in the list on purpose:
    ``tests/test_wfn_loader_eager.py`` pins ``_phdf5_unfold_kernel``'s
    output against the eager backend, which is why it is door surface
    rather than an implementation detail.

    ``_build_phdf5_clamped_counts`` was the name asserted here until
    2026-08-07; it is now ``file_io._slab_io_ffi._derive_window_counts``,
    behind the slab_io door, and the cells that read it moved with it.
    The named assertion moved to the kernel that stayed rather than being
    dropped: without one, this cell degrades to "``__all__`` is
    self-consistent", which a door that re-exported only ``WfnLoader``
    would also satisfy.
    """
    _needs_deps()
    run = import_isolation(
        "wfn_loader", _lorrax_roots(), src_dir=_SVC_SRC, deps=_DEPS,
        extra_path=[_LXKIT_SRC], check_path=True,
        preamble="import wfn_loader as _w\n"
                 "for _n in _w.__all__:\n"
                 "    assert getattr(_w, _n, None) is not None, _n\n"
                 "assert _w.WfnLoader.__name__ == 'WfnLoader'\n"
                 "assert '_phdf5_unfold_kernel' in _w.__all__\n"
                 "assert _w.KSpec is not None\n")
    assert run.loaded == ()


def test_wfn_loader_still_imports_clean_with_lorrax_on_the_path():
    """Isolation must not be an artifact of the monorepo being unreachable:
    with lorrax's src right there, wfn_loader still touches none of it.

    This is the cell the lazy imports are really about.  ``common.
    gvec_fft_box``, ``file_io.mf_header`` and ``file_io.slab_io`` are
    all IMPORTABLE here — so a module-scope import of any of them would
    succeed and show up in ``loaded``, which is precisely what the
    monorepo-absent cell above cannot distinguish from a typo in a path.

    The ``reachable`` assertion is what keeps this cell honest — without it
    a typo in the extra path would make the leg pass by measuring the
    isolated case twice.
    """
    _needs_monorepo()
    _needs_deps()
    run = import_isolation("wfn_loader", _lorrax_roots(), src_dir=_SVC_SRC,
                           deps=_DEPS,
                           extra_path=[_LXKIT_SRC, _LORRAX_SRC],
                           check_path=False)
    assert run.loaded == ()
    assert {r for r, _ in run.reachable} >= {"file_io", "ffi", "common"}, (
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
    _needs_deps()
    with pytest.raises(AssertionError, match="pulled"):
        import_isolation("wfn_loader", _lorrax_roots(), src_dir=_SVC_SRC,
                         deps=_DEPS, extra_path=[_LXKIT_SRC, _LORRAX_SRC],
                         check_path=False, preamble="import ffi")


def test_the_wrong_copy_of_the_package_is_a_failure():
    """``src_dir`` is the thing under test; resolving the import somewhere
    else (an installed wheel, a stale build dir) measures the wrong file."""
    _needs_deps()
    with pytest.raises(AssertionError, match="not under the src dir"):
        import_isolation("wfn_loader", ("gw",),
                         src_dir=os.path.join(_REPO, "no_such_src"),
                         deps=_DEPS, extra_path=[_SVC_SRC, _LXKIT_SRC],
                         check_path=False)

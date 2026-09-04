"""Standalone, falsifiably — and HONESTLY SPLIT, which is this service's case.

``distrib_la``'s claim is simple: nothing of lorrax, ever.  ``zeta_loader``'s
is not, and pretending otherwise would be the lie the charter's standalone
criterion exists to catch.  Design D8 states the split out loud, so this file
measures BOTH halves of it:

* ``import zeta_loader`` drags in NO lorrax package.  Clean, with the
  monorepo unreachable in both senses (``sys.modules`` AND ``sys.path``).
* The FORMAT surface — ``probe_zeta_file`` — is FULLY
  FUNCTIONAL there.  Not "importable": a probe that imports and then cannot
  read a file is not a standalone probe.  Both of its production call sites
  run OUTSIDE a loader's lifetime (the two ``gw_init`` probes gate whether
  the fit runs at all), so this is the half that has to work with nothing
  else present.
* The DATA path is NOT standalone, and it REFUSES BY NAME rather than
  dying in an ImportError traceback three frames deep. The application seam
  is observable: ``file_io.mf_header``,
  ``file_io.isdf_header``, ``file_io.slab_io`` and ``common.gvec_fft_box``
  are call-time imports with named refusals that route callers to the
  canonical application closure rather than teaching a second path scan.

``zeta_loader`` DOES import jax (every data read takes a
``jax.sharding.Mesh`` and returns a sharded ``jax.Array``) and h5py (the ζ
file is HDF5 and the serial read of its two header groups is what makes
``mesh=None`` work on a stack with no phdf5 FFI).  Both are handed to the
child on purpose, BY NAME.

THE ``python -S`` LESSON, measured, not defensive: this repo's venv carries
an editable-install ``.pth``, so an ordinary subprocess of the test
interpreter has ``<tree>/src`` on ``sys.path`` no matter what ``PYTHONPATH``
says.  ``-S`` skips ``site`` and therefore every ``.pth``; the child gets its
dependencies back BY NAME, as DIRECTORIES, which carry no ``.pth``
processing.  :func:`lxkit.testing.import_isolation` does all of that.
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

#: zeta_loader's declared runtime dependencies, handed to the child BY NAME.
#: Naming them is the claim; anything not named — nor in a named package's own
#: declared requirements, which ``dep_dirs`` follows transitively — must stay
#: unreachable.  h5py is on this list and ``file_io.slab_io`` is not, which is
#: the pyproject's quarantine statement restated as a test input: the SERIAL
#: half of the I/O is this package's own declared native dep, the PARALLEL
#: half is reached through the host tree and is not linked here.
_DEPS = ("jax", "h5py", "numpy")

#: What zeta_loader must not touch.  Derived from the monorepo when it is
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

def test_zeta_loader_imports_with_the_monorepo_absent():
    """The charter criterion.  lxkit, jax, h5py and numpy are on the child's
    path — they are declared dependencies — and lorrax is not, in either
    sense.

    ``check_path=True``: BOTH halves.  ``sys.modules`` proves zeta_loader
    did not IMPORT lorrax; ``sys.path`` proves it COULD not have, and the
    second is what makes the first evidence about the package rather than
    about this machine.
    """
    _needs_jax()
    run = import_isolation("zeta_loader", _lorrax_roots(), src_dir=_SVC_SRC,
                           deps=_DEPS, extra_path=[_LXKIT_SRC],
                           check_path=True)
    assert run.file.startswith(_SVC_SRC + os.sep)
    assert run.loaded == () and run.reachable == ()


def test_the_format_surface_is_fully_functional_with_no_lorrax():
    """``probe_zeta_file`` runs in the isolated child.

    "Importable" is the weak claim and it is not the one D8 makes.  Both of
    This function has production call sites that run where no loader could
    exist, so the standalone property has to be about what it DOES:

    * the probe answers on a garbage file (``readable=False`` + an error
      record) and on ``None`` — its never-raises contract, exercised where
      there is no lorrax to fall back on.

    The fixture builder (``zeta_synth``) is on the child's path and is pure
    h5py+numpy for exactly this reason — see its module docstring.  A
    builder that imported ``file_io`` could not run here, and this leg would
    have had to assert something weaker.
    """
    _needs_jax()
    preamble = (
        "import os, tempfile\n"
        "import numpy as np\n"
        "import zeta_synth as Z\n"
        "import zeta_loader as ZL\n"
        "tmp = tempfile.mkdtemp()\n"
        # --- probe: garbage in, an error RECORD out, never an exception ---
        "junk = os.path.join(tmp, 'junk.h5')\n"
        "open(junk, 'wb').write(b'this is not HDF5. ' * 200)\n"
        "p = ZL.probe_zeta_file(junk)\n"
        "assert p.exists is True and p.readable is False and p.error\n"
        "assert p.dataset_name is None and p.zeta_done is None\n"
        "assert ZL.probe_zeta_file(None).readable is False\n"
        "assert ZL.probe_zeta_file(os.path.join(tmp, 'nope.h5')).exists"
        " is False\n"
        # --- probe: a REAL zeta file, read correctly, with no lorrax ------
        "z = os.path.join(tmp, 'zeta_q.h5')\n"
        "Z.build_gflat(z, n_q=3, n_rmu=5, ngkmax=4)\n"
        "q = ZL.probe_zeta_file(z)\n"
        "assert q.readable is True and q.dataset_name == 'zeta_q_G'\n"
        "assert q.mu_extent == 5 and q.zeta_done is True\n"
        "assert q.r_mu_fft_idx.shape == (5, 3)\n"
    )
    run = import_isolation("zeta_loader", _lorrax_roots(), src_dir=_SVC_SRC,
                           deps=_DEPS, extra_path=[_LXKIT_SRC, _TESTS],
                           check_path=True, preamble=preamble)
    assert run.loaded == () and run.reachable == ()


def test_the_format_surface_needs_no_jax():
    """The whole format surface runs before jax is ever imported — and
    touching the reader is exactly what imports it.

    Measured, not hypothetical (step-4 lesson): the first login-node
    diagnostic against a production ζ file could not even ``import
    zeta_loader`` to reach the probe, because the loader's module-scope
    jax import came in through ``__init__``.  Login-node ``python3`` has
    numpy + h5py and NO jax, and the probe/append pair is pure h5py+numpy
    — so the door now defers the loader (PEP 562) and this cell pins both
    halves: the format surface leaves ``sys.modules`` jax-free, and the
    ``ZetaLoader`` attribute is still served (its access is what pays the
    jax import, asserted here on a stack that has jax so the positive
    control can run).
    """
    _needs_jax()
    preamble = (
        "import sys, os, tempfile\n"
        "import numpy as np\n"
        "import zeta_loader as ZL\n"
        "assert 'jax' not in sys.modules, 'import zeta_loader pulled jax'\n"
        "tmp = tempfile.mkdtemp()\n"
        "z = os.path.join(tmp, 'zeta_q.h5')\n"
        "import zeta_synth as Z\n"
        "Z.build_gflat(z, n_q=2, n_rmu=3, ngkmax=4)\n"
        "p = ZL.probe_zeta_file(z)\n"
        "assert p.readable and p.dataset_name == 'zeta_q_G'\n"
        "assert 'jax' not in sys.modules, 'the format surface pulled jax'\n"
        # the positive control: the reader is lazy, not gone.
        "cls = ZL.ZetaLoader\n"
        "assert cls.__name__ == 'ZetaLoader'\n"
        "assert 'jax' in sys.modules, 'ZetaLoader access should import jax'\n"
    )
    run = import_isolation("zeta_loader", _lorrax_roots(), src_dir=_SVC_SRC,
                           deps=_DEPS, extra_path=[_LXKIT_SRC, _TESTS],
                           check_path=True, preamble=preamble)
    assert run.loaded == () and run.reachable == ()


def test_the_data_path_refuses_by_naming_the_missing_host_tree_module():
    """THE LAZY-REFUSAL LEG.  A REAL ζ file, and a construction that refuses.

    This is the half D8 could most easily have faked: "the data path needs
    lorrax" is satisfiable by an ImportError from anywhere, and an
    ImportError from anywhere is exactly what the seam is supposed to
    replace.  So the child builds a genuine, well-formed ``zeta_q.h5`` with
    raw h5py — a file a ``ZetaLoader`` would open happily in the monorepo —
    and then asserts that constructing one with lorrax absent produces the
    NAMED refusal ``loader._host_tree_refusal`` writes, not a bare import
    failure.

    ``file_io.mf_header`` is the module named because ``_mf_header_binders``
    is the FIRST of the four helpers ``__init__`` calls; the four share one
    refusal sentence precisely so they cannot drift apart.
    """
    _needs_jax()
    preamble = (
        "import os, tempfile\n"
        "import zeta_synth as Z\n"
        "import zeta_loader as ZL\n"
        "tmp = tempfile.mkdtemp()\n"
        "z = os.path.join(tmp, 'zeta_q.h5')\n"
        "Z.build_gflat(z, n_q=2, n_rmu=3, ngkmax=4)\n"
        # The file is REAL: the standalone probe reads it fine, so the
        # refusal below is about the host tree and not about the file.
        "assert ZL.probe_zeta_file(z).dataset_name == 'zeta_q_G'\n"
        "try:\n"
        "    ZL.ZetaLoader(z)\n"
        "except ImportError as exc:\n"
        "    m = str(exc)\n"
        "    assert 'file_io.mf_header' in m, m\n"
        "    assert 'bind_mf_attrs' in m, m\n"
        "    assert 'HOST-TREE module' in m, m\n"
        "    assert 'runtime.source_closure.ensure_source_closure()' in m, m\n"
        "    assert 'compatibility delegate' in m, m\n"
        "    assert 'do not append individual service roots' in m, m\n"
        "    assert 'probe_zeta_file' in m, m\n"
        "else:\n"
        "    raise AssertionError(\n"
        "        'ZetaLoader constructed with lorrax off sys.path; the "
        "lazy-import seam is not a seam')\n"
    )
    run = import_isolation("zeta_loader", _lorrax_roots(), src_dir=_SVC_SRC,
                           deps=_DEPS, extra_path=[_LXKIT_SRC, _TESTS],
                           check_path=True, preamble=preamble)
    assert run.loaded == () and run.reachable == ()


def test_the_four_lazy_helpers_share_one_refusal_sentence():
    """The in-process pin of what the subprocess leg observes ONCE.

    ``loader._host_tree_refusal`` exists so the four call-time importers
    cannot drift apart: each names its own module, its own symbols and its
    own "needed for", and everything else — the HOST-TREE explanation, the
    canonical closure, its compatibility delegate, and the pointer at the
    surface that still works — is one string.  Asserting that here means the
    subprocess only has to prove the sentence FIRES, not that all four
    spellings of it are complete.
    """
    from zeta_loader.loader import _host_tree_refusal
    modules = {
        "file_io.mf_header": "bind_mf_attrs / read_mf_header_from_file",
        "file_io.isdf_header": "bind_isdf_attrs / read_isdf_header_from_file",
        "file_io.slab_io": "SlabIO",
        "common.gvec_fft_box":
            "fft_box_pad_sentinel / pad_gvecs_to_sentinel",
    }
    for mod, names in modules.items():
        msg = _host_tree_refusal(mod, names, "some surface")
        assert mod in msg and names in msg
        assert "HOST-TREE module" in msg
        assert "runtime.source_closure.ensure_source_closure()" in msg
        assert "compatibility delegate" in msg
        assert "do not append individual service roots" in msg
        assert "probe_zeta_file" in msg
        assert "pure h5py+numpy" in msg


def test_zeta_loader_still_imports_clean_with_lorrax_on_the_path():
    """Isolation must not be an artifact of the monorepo being unreachable:
    with lorrax's src right there, ``import zeta_loader`` still touches none
    of it.

    The ``reachable`` assertion is what keeps this cell honest — without it
    a typo in the extra path would make the leg pass by measuring the
    isolated case twice.  It matters more here than for distrib_la: this
    package NAMES four lorrax modules in its own source, so "did not import
    them" has to be measured with them sitting right there importable.
    """
    _needs_monorepo()
    _needs_jax()
    run = import_isolation("zeta_loader", _lorrax_roots(), src_dir=_SVC_SRC,
                           deps=_DEPS,
                           extra_path=[_LXKIT_SRC, _LORRAX_SRC],
                           check_path=False)
    assert run.loaded == ()
    assert {r for r, _ in run.reachable} >= {"gw", "file_io", "common"}, (
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
        import_isolation("zeta_loader", _lorrax_roots(), src_dir=_SVC_SRC,
                         deps=_DEPS,
                         extra_path=[_LXKIT_SRC, _LORRAX_SRC],
                         check_path=False, preamble="import ffi")


def test_importing_the_host_tree_module_this_package_names_is_still_a_leak():
    """The twin with the module this service ACTUALLY reaches for.

    ``import ffi`` proves the mechanism.  ``import file_io.isdf_header``
    proves the mechanism catches the specific leak that would matter here:
    someone promoting one of the four lazy call-time imports to module
    scope, which is the single most likely way this package stops being
    standalone.
    """
    _needs_monorepo()
    _needs_jax()
    with pytest.raises(AssertionError, match="pulled"):
        import_isolation("zeta_loader", _lorrax_roots(), src_dir=_SVC_SRC,
                         deps=_DEPS,
                         extra_path=[_LXKIT_SRC, _LORRAX_SRC],
                         check_path=False,
                         preamble="import file_io.isdf_header")


def test_the_wrong_copy_of_the_package_is_a_failure():
    """``src_dir`` is the thing under test; resolving the import somewhere
    else (an installed wheel, a stale build dir) measures the wrong file."""
    _needs_jax()
    with pytest.raises(AssertionError, match="not under the src dir"):
        import_isolation("zeta_loader", ("gw",),
                         src_dir=os.path.join(_REPO, "no_such_src"),
                         deps=_DEPS, extra_path=[_SVC_SRC, _LXKIT_SRC],
                         check_path=False)


def test_the_standalone_surface_check_can_fail():
    """RED TWIN for the FORMAT-surface leg itself.

    The functional legs above run their assertions inside the child, where a
    failure surfaces as a non-zero exit and no payload line.  If that
    plumbing did not work, every ``preamble`` assertion in this file would
    be decorative.  This asserts a deliberately FALSE claim in the child and
    demands that ``import_isolation`` raise.
    """
    _needs_jax()
    with pytest.raises(AssertionError):
        import_isolation(
            "zeta_loader", _lorrax_roots(), src_dir=_SVC_SRC, deps=_DEPS,
            extra_path=[_LXKIT_SRC, _TESTS], check_path=True,
            preamble=("import zeta_loader as ZL\n"
                      "assert ZL.probe_zeta_file(None).readable is True\n"))

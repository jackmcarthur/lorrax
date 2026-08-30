"""Standalone, falsifiably — the quarantine this service exists to have.

``symmetry_maps`` absorbed three modules that between them used to reach
``file_io`` (a dead ``from file_io import WfnLoader``, the only real cycle
with the loader) and ``psp`` (``density_symmetry_check``'s valence-density
quadrature).  Both edges are gone by construction rather than by discipline:
the dead import was DELETED in the extraction commit (design decision 12)
and the quadrature is INJECTED (``valence_density_fn`` /
``spin_degeneracy_fn``, WAVE1_BRIEF ruling 4) with a ``None`` default that
does the same lazy ``psp.get_DFT_mtxels`` import the module always did, on
the one call path that has a loader in hand.

"By construction" is a claim, and the only way to observe it is a process
where lorrax is ABSENT.  :func:`lxkit.testing.import_isolation` builds one:
``python -S`` (so this venv's ``__editable__.lorrax-0.1.0.pth`` cannot put
``<tree>/src`` on the child's path behind our back), ``PYTHONPATH`` set to
exactly the service's ``src`` plus its DECLARED dependencies resolved by
name, and a cwd outside the repo.  Both halves are asserted: ``sys.modules``
says the package did not IMPORT lorrax, ``sys.path`` says it COULD not
have — and the second is what makes the first evidence about the package
rather than about this machine.

WHAT IS HANDED TO THE CHILD, AND WHY.  ``pyproject.toml`` declares
``lxkit``, ``jax`` and ``numpy``.  jax is not a concession:
``unfold_isdf_operator``
is a ``shard_map`` over an ``('x','y')`` mesh and ``maps.py`` imports it at
module scope.  Anything NOT declared — ``h5py`` above all, which this
suite's own deck tier needs and the PACKAGE does not — must stay
unreachable, and :func:`test_the_package_answers_table_questions_with_no_h5py`
is where that is pinned.
"""

from __future__ import annotations

import os
import shutil

import pytest

from lxkit.testing import import_isolation

_TESTS = os.path.dirname(os.path.abspath(__file__))
_SVC_SRC = os.path.join(os.path.dirname(_TESTS), "src")
_SERVICES = os.path.dirname(os.path.dirname(_TESTS))
_LXKIT_SRC = os.path.join(_SERVICES, "lxkit", "src")
_REPO = os.path.dirname(_SERVICES)
_LORRAX_SRC = os.path.join(_REPO, "src")

#: The declared runtime dependencies, handed to the child BY NAME.
#: ``lxkit`` arrives through ``extra_path`` instead because it is an
#: in-tree sibling with no wheel — the same shape distrib_la uses.
_DEPS = ("jax", "numpy")

#: The lorrax roots this service must not touch, as a fallback for a
#: checkout with no ``src/``.  ``file_io``, ``psp`` and ``common`` are the
#: three the survey named: the dead loader import, the quadrature, and the
#: three modules' old home.
_CORE = ("bandstructure", "bse", "centroid", "common", "ffi", "file_io",
         "gw", "isdf", "postprocess", "psp", "runtime", "solvers")


def _lorrax_roots() -> tuple[str, ...]:
    """Derived from the monorepo when it is there, so a new top-level
    lorrax package is covered the day it lands rather than the day someone
    remembers to add it to a literal.

    NAMESPACE PACKAGES COUNT.  ``src/psp/`` has NO ``__init__.py`` — it is
    an implicit namespace package — and a roots list filtered on
    ``__init__.py`` (which is how ``services/distrib_la``'s copy of this
    helper is written) drops it silently.  psp is ruling 4's whole subject:
    the quadrature injection exists so ``density_symmetry_check`` does not
    import it at module scope, and a quarantine test that never named the
    package it is quarantining would be the tautology this suite is full of
    cells against.  Any directory holding a ``.py`` is a root here.
    """
    if os.path.isdir(_LORRAX_SRC):
        found = []
        for n in sorted(os.listdir(_LORRAX_SRC)):
            d = os.path.join(_LORRAX_SRC, n)
            if not os.path.isdir(d):
                continue
            if os.path.isfile(os.path.join(d, "__init__.py")) or any(
                    f.endswith(".py") for f in os.listdir(d)):
                found.append(n)
        if found:
            return tuple(found)
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

def test_symmetry_maps_imports_with_the_monorepo_absent():
    """The charter criterion, both halves.

    ``run.loaded == ()`` — no lorrax package entered ``sys.modules``.
    ``run.reachable == ()`` — none was even importable, so the first is a
    statement about ``symmetry_maps`` and not about a machine that happens
    to lack a checkout.
    """
    _needs_jax()
    run = import_isolation("symmetry_maps", _lorrax_roots(), src_dir=_SVC_SRC,
                           deps=_DEPS, extra_path=[_LXKIT_SRC],
                           check_path=True)
    assert run.file.startswith(_SVC_SRC + os.sep)
    assert run.loaded == () and run.reachable == ()


def test_the_three_absorbed_modules_all_import_clean():
    """``import symmetry_maps`` is the door, but the modules are the risk.

    The package ``__init__`` re-exports from all three; a module that
    reached ``file_io`` or ``psp`` only under a submodule import would slip
    past a cell that imported the top level and stopped.  Named
    individually so a failure says WHICH module regressed.
    """
    _needs_jax()
    run = import_isolation(
        "symmetry_maps", _lorrax_roots(), src_dir=_SVC_SRC, deps=_DEPS,
        extra_path=[_LXKIT_SRC], check_path=True,
        preamble="import symmetry_maps.maps\n"
                 "import symmetry_maps.orbit_syms\n"
                 "import symmetry_maps.density_symmetry_check\n"
                 "import symmetry_maps._shard_map\n")
    assert run.loaded == ()


def test_the_injected_quadrature_keeps_psp_out_of_the_import_graph():
    """WAVE1_BRIEF ruling 4, observed rather than described.

    ``check_density_symmetries`` has keyword-only ``valence_density_fn``,
    ``spin_degeneracy_fn`` and ``spin_density_fields_fn`` parameters whose
    ``None`` defaults preserve lazy ``psp.get_DFT_mtxels`` imports.  The
    claim is that BINDING the
    function — reading its signature, which is what a caller wiring the
    injection does — costs no psp import.  Asserted in the isolated child,
    where a psp import would be an ``ImportError`` and not a silent success.
    """
    _needs_jax()
    run = import_isolation(
        "symmetry_maps", _lorrax_roots(), src_dir=_SVC_SRC, deps=_DEPS,
        extra_path=[_LXKIT_SRC], check_path=True,
        preamble="import inspect\n"
                 "import symmetry_maps as S\n"
                 "p = inspect.signature(S.check_density_symmetries).parameters\n"
                 "assert 'valence_density_fn' in p, sorted(p)\n"
                 "assert 'spin_degeneracy_fn' in p, sorted(p)\n"
                 "assert 'spin_density_fields_fn' in p, sorted(p)\n"
                 "assert p['valence_density_fn'].default is None\n"
                 "assert p['spin_density_fields_fn'].default is None\n"
                 "assert p['valence_density_fn'].kind is "
                 "inspect.Parameter.KEYWORD_ONLY\n"
                 "assert p['spin_density_fields_fn'].kind is "
                 "inspect.Parameter.KEYWORD_ONLY\n")
    assert run.loaded == ()


def test_the_package_answers_table_questions_without_importing_h5py():
    """h5py is a dependency of this SUITE, never of the package.

    ``SymMaps`` reads eleven attributes off whatever object it is handed;
    the star helpers take four arrays.  Nothing in the package opens a
    file — which is what makes the L-a tier possible at all, and what makes
    the deck tier's stub a legitimate stand-in for ``WfnLoader``.

    REACHABLE AND NOT LOADED, in that order.  ``deps=`` resolves jax and
    numpy to this venv's ``site-packages``, so everything installed beside
    them — h5py included — is on the child's path whether we want it or
    not.  Asserting "h5py cannot be imported" here would be a claim about
    the venv layout.  The claim that IS about the package is that h5py is
    right there and ``import symmetry_maps`` still never touches it, so the
    cell asserts the availability FIRST and the absence from
    ``sys.modules`` second.  The table it builds is gnppm's real
    ``(irr_idx_k, sym_idx_k)``, including the non-monotone first-occurrence
    labels ``[0, 2, 6, 8, 7]``.
    """
    _needs_jax()
    run = import_isolation(
        "symmetry_maps", _lorrax_roots(), src_dir=_SVC_SRC, deps=_DEPS,
        extra_path=[_LXKIT_SRC], check_path=True,
        preamble="import importlib.util, sys\n"
                 "assert importlib.util.find_spec('h5py') is not None, (\n"
                 "    'h5py is not even on this path, so not importing it "
                 "proves nothing')\n"
                 "import numpy as np\n"
                 "import symmetry_maps as S\n"
                 "irr = np.array([0,2,2,6,8,7,6,7,8], dtype=np.int32)\n"
                 "sid = np.array([0,2,0,2,2,2,0,0,0], dtype=np.int32)\n"
                 "km = S.KStarMap(irr, sid, 2)\n"
                 "assert km.nk_irr == 5 and km.labels.tolist() == "
                 "[0,2,6,8,7]\n"
                 "assert 'h5py' not in sys.modules, (\n"
                 "    'symmetry_maps imported h5py; the package answers "
                 "table questions from arrays')\n")
    assert run.loaded == ()


def test_symmetry_maps_still_imports_clean_with_lorrax_on_the_path():
    """Isolation must not be an artifact of the monorepo being unreachable.

    With lorrax's ``src/`` right there on the child's path, the package
    still touches none of it.  The ``reachable`` assertion is what keeps
    this honest: without it a typo in the extra path would make the cell
    pass by measuring the isolated case a second time.
    """
    _needs_monorepo()
    _needs_jax()
    run = import_isolation("symmetry_maps", _lorrax_roots(), src_dir=_SVC_SRC,
                           deps=_DEPS,
                           extra_path=[_LXKIT_SRC, _LORRAX_SRC],
                           check_path=False)
    assert run.loaded == ()
    assert {r for r, _ in run.reachable} >= {"file_io", "common", "gw"}, (
        f"lorrax was supposed to be ON the path here; saw {run.reachable}")
    # psp is FORBIDDEN but can never appear in ``reachable``: lxkit's probe
    # looks for ``<root>/__init__.py`` or ``<root>.py`` and ``src/psp/`` is
    # an implicit NAMESPACE package with neither.  So the psp half of the
    # quarantine rests on ``loaded`` alone, and the roots list is where that
    # has to be checked — a psp dropped from the list would make every cell
    # in this file green about a package nobody was watching.
    assert "psp" in _lorrax_roots(), (
        "psp is not in the forbidden roots; ruling 4's quadrature injection "
        "would be unwatched because src/psp has no __init__.py")


# ---------------------------------------------------------------------------
# RED TWINS — the check, shown failing
# ---------------------------------------------------------------------------

def test_the_isolation_check_can_fail():
    """A deliberate lorrax import MUST break the check.

    ``import ffi`` is the probe because ``src/ffi/__init__.py`` is
    stdlib-only at module scope, so the twin fails on the LEAK rather than
    on an unrelated ImportError from a package that wanted numpy-of-a-
    different-version.
    """
    _needs_monorepo()
    _needs_jax()
    with pytest.raises(AssertionError, match="pulled"):
        import_isolation("symmetry_maps", _lorrax_roots(), src_dir=_SVC_SRC,
                         deps=_DEPS,
                         extra_path=[_LXKIT_SRC, _LORRAX_SRC],
                         check_path=False, preamble="import ffi")


def test_the_dead_file_io_import_would_be_caught():
    """The SPECIFIC edge decision 12 deleted, shown as a failure.

    ``maps.py`` carried ``from file_io import WfnLoader`` at module scope
    with no user — the one real cycle with the loader and the whole
    import-isolation blocker.  This is what its return would look like from
    here: the leak is reported, and it names ``file_io``.
    """
    _needs_monorepo()
    _needs_jax()
    with pytest.raises(AssertionError, match="pulled") as ei:
        import_isolation("symmetry_maps", _lorrax_roots(), src_dir=_SVC_SRC,
                         deps=_DEPS,
                         extra_path=[_LXKIT_SRC, _LORRAX_SRC],
                         check_path=False,
                         preamble="from file_io import WfnLoader")
    assert "file_io" in str(ei.value)


def test_a_psp_import_would_be_caught_even_though_psp_is_a_namespace_package():
    """RED TWIN for the half ``reachable`` structurally cannot see.

    ``src/psp/`` has no ``__init__.py``, so lxkit's path probe never
    reports it and the psp quarantine rests entirely on the ``sys.modules``
    half.  This is that half, shown firing: an eager
    ``import psp.get_DFT_mtxels`` — the exact module ruling 4's injection
    exists to keep lazy — must make the run refuse and name psp.
    """
    _needs_monorepo()
    _needs_jax()
    with pytest.raises(AssertionError, match="pulled") as ei:
        import_isolation("symmetry_maps", _lorrax_roots(), src_dir=_SVC_SRC,
                         deps=_DEPS,
                         extra_path=[_LXKIT_SRC, _LORRAX_SRC],
                         check_path=False,
                         preamble="import psp.get_DFT_mtxels")
    assert "psp" in str(ei.value)


def test_the_wrong_copy_of_the_package_is_a_failure():
    """``src_dir`` is the thing under test; resolving the import somewhere
    else (an installed wheel, a stale build dir, the old
    ``common/symmetry_maps.py`` shim) measures the wrong file."""
    _needs_jax()
    with pytest.raises(AssertionError, match="not under the src dir"):
        import_isolation("symmetry_maps", ("gw",),
                         src_dir=os.path.join(_REPO, "no_such_src"),
                         deps=_DEPS, extra_path=[_SVC_SRC, _LXKIT_SRC],
                         check_path=False)


# ---------------------------------------------------------------------------
# IMPORTABLE WITHOUT A DEVICE.  jax as a DEPENDENCY, not as a backend.
# ---------------------------------------------------------------------------
# The cells above hand the child jax and get a working default backend for
# free, so none of them can tell "the package imports" apart from "the
# package imports AND a device happened to be there".  On 2026-08-08 that
# distinction cost the whole file: on Perlmutter the stripped child inherits
# the session's ``JAX_PLATFORMS`` (which names ``cuda``) and cannot reach the
# CUDA plugin from a ``python -S`` path, so the first device touch on the
# import path raised ``Unable to initialize backend 'cuda'`` and all nine
# cells above went red.  The touch was
# ``orbit_syms._CANON_INV = jnp.int64(10**12)`` at module scope, reached from
# the door through ``qirr_store``; it is a numpy scalar now.
#
# WHY A BACKEND NAME NOTHING PROVIDES, rather than ``cuda``.  ``cuda`` is a
# statement about the machine: jax SKIPS it quietly on a host with no visible
# NVIDIA GPU (``xla_bridge`` guards the registration on
# ``has_visible_nvidia_gpu()``), so a WSL box cannot reproduce the Perlmutter
# red with it and the cell would be green for the wrong reason — which is
# exactly how the WSL census reported zero newly red while Perlmutter had
# nine.  An unknown name is refused by EVERY jax on EVERY machine, at the
# first device touch and never at ``import jax``, so it asks the one
# portable question: does the import path touch a device at all?

_NO_SUCH_BACKEND = "lorrax_no_such_backend"


@pytest.fixture()
def _no_usable_backend(monkeypatch):
    """The child inherits ``os.environ``; this is how it gets the pin."""
    monkeypatch.setenv("JAX_PLATFORMS", _NO_SUCH_BACKEND)
    # x64 for the same reason production sets it: the canonicalisation keys
    # are int64 and a child left in x32 would fail on the DTYPE instead,
    # which is a different sentence and would mask this one.
    monkeypatch.setenv("JAX_ENABLE_X64", "1")


def test_importing_the_package_initialises_no_jax_backend(_no_usable_backend):
    """The door is importable on a machine with no usable device.

    A table library needs jax to COMPUTE and must not need it to LOAD.  The
    whole public surface is bound here — the door, all three absorbed
    modules and the shard-map helper — under a ``JAX_PLATFORMS`` no backend
    can satisfy, so anything that materialises an array or asks for a
    default device while a module body runs fails this cell by name.
    """
    _needs_jax()
    run = import_isolation(
        "symmetry_maps", _lorrax_roots(), src_dir=_SVC_SRC, deps=_DEPS,
        extra_path=[_LXKIT_SRC], check_path=True,
        preamble="import symmetry_maps.maps\n"
                 "import symmetry_maps.orbit_syms\n"
                 "import symmetry_maps.density_symmetry_check\n"
                 "import symmetry_maps.qirr_store\n"
                 "import symmetry_maps._shard_map\n")
    assert run.loaded == ()


def test_a_module_scope_device_array_would_be_caught(_no_usable_backend, tmp_path):
    """RED TWIN for the cell above — the exact edge, put back.

    A COPY of the service tree with one line appended to ``orbit_syms``:
    ``jnp.int64(10**12)`` at module scope, which is what stood at
    ``orbit_syms.py:246`` until this branch.  Nothing else differs, and the
    copy is what ``src_dir`` points at, so a green cell above cannot be
    green because the check stopped working.

    The refusal is the harness's own ``produced no LXKIT_ISOLATION line``:
    the child dies before it can print, which is the shape the Perlmutter
    census recorded nine times.
    """
    _needs_jax()
    twin_src = tmp_path / "twin_src"
    shutil.copytree(_SVC_SRC, twin_src)
    with open(twin_src / "symmetry_maps" / "orbit_syms.py", "a") as fh:
        fh.write("\n\n_EAGER_TWIN = jnp.int64(10**12)\n")
    with pytest.raises(AssertionError, match="produced no LXKIT_ISOLATION"):
        import_isolation("symmetry_maps", _lorrax_roots(),
                         src_dir=str(twin_src), deps=_DEPS,
                         extra_path=[_LXKIT_SRC], check_path=True)

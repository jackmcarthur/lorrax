"""Every service door is reachable in a BARE launch, not just under pytest.

THE DEFECT THIS EXISTS TO CATCH is "green suite, red cluster".  Nothing in
the launch chain puts ``services/*/src`` on ``sys.path``: ``lx`` rewrites
the container ``PYTHONPATH`` to exactly ``<checkout>/src``
(``~/bin/lx``, ``retarget_pythonpath``), the Shifter image pip-installs
nothing, and ``tests/harness.run_gw_jax`` — the launcher every end-to-end
regression gate goes through — sets ``PYTHONPATH`` to ``<repo>/src`` and
nothing else.  Under pytest the path is there anyway, because each
``services/*/tests/conftest.py`` inserts it at collection.  So a
module-scope ``from distrib_la import ...`` — or ``from wfn_loader import
...`` — with no bootstrap would pass the entire lorrax suite and
``ImportError`` on the first real run.

PARAMETERIZED BY ``(consumer, service)`` since 2026-08-07.  It was
distrib_la-only, and the wfn_loader extraction added a second service
whose transitional SHIM (``src/file_io/wfn_loader.py``) is a module-scope
bootstrap consumer: the same defect class, one service later (charter
wave 1, step-1a adjudication (i)).

THE wfn_loader AND symmetry_maps BRANCHES BOTH GENERALIZED THIS FILE, in
parallel and without seeing each other, and the 2026-08-08 landing merge
kept BOTH generalizations because they measure different things.  From
wfn_loader: the bootstrap-ORDER census (``_scan_tree``, door-parameterized
already), which asserts every module-scope door import has an
``ensure_on_path()`` call strictly above it.  From symmetry_maps: the
BUCKETED census, which asserts the set of module-scope consumers equals
the union of four lists, so a new consumer has to be classified —
bare-launch-covered, FFI-blocked, shim, or script — before the suite goes
green.  Dropping either would have silently retired a real gate; the
merge added the symmetry_maps door to the first and left the second
scoped to its own door, which is what its own comment asks for.

``ffi._services.ensure_on_path()`` is the bootstrap, and it is transitional
plumbing with an owner decision behind it (see its docstring: pip install -e
services/*, a modulefile PYTHONPATH entry, or a uv workspace — all touch
shared resources).  These cells are what keeps it honest until then.

Each cell launches a SUBPROCESS with ``PYTHONPATH=<repo>/src`` and nothing
else, which is the cluster's environment exactly, and imports a consumer
that reaches the door at MODULE scope.

RED ARM: ``test_a_bare_launch_cannot_find_the_service_by_itself`` shows the
same subprocess FAILING to import each service directly.  Without it these
cells could be passing because the service happens to be installed, and
would say nothing about the bootstrap at all.  It runs once per SERVICE:
a box with an editable install of one of them would make only that
service's consumer cells vacuous, and a single red arm naming the other
would report everything as fine.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SRC = str(_REPO / "src")

#: ``(consumer module, service it must reach)`` — consumers that import a
#: service door at MODULE scope.  The lazy sites (seven of them, all
#: ``ensure_on_path()`` inside the function) cannot fail at import time and
#: are covered by the end-to-end run instead.
#:
#: GENERALIZED 2026-08-07 from a bare tuple of distrib_la consumers.  It
#: had to be: ``file_io.wfn_loader`` became a module-scope bootstrap
#: consumer with the wfn_loader extraction, and a parametrization that
#: could only name one service would have left it with NO bare-launch cell
#: at all — which is the "green suite, red cluster" class this whole file
#: exists for, one service later (charter wave 1, step-1a adjudication (i)).
#: ``file_io.wfn_loader`` is the TRANSITIONAL SHIM: every in-tree spelling
#: of ``from file_io.wfn_loader import WfnLoader`` goes through it, so it
#: is exactly the import a cluster run performs first, and the one that
#: would ``ImportError`` on a bootstrap that stopped working.
#:
#: ``bandstructure.bse_setup`` is a third module-scope consumer and is
#: deliberately NOT here: it runs ``initialize_communicator_stack()`` at
#: import (through ``.htransform``), which REQUIRES an FFI ``.so``, so a
#: cell naming it would be a skip on every machine without one — the exact
#: shape of coverage that evaporates quietly.  The Si COHSEX fixture run
#: covers it on the cluster, where the bootstrap is what the whole gate
#: depends on.
#: ``psp.operator_checks`` is the step-3 replumb's representative: that
#: sweep turned ~14 lorrax modules into module-scope consumers of the
#: ``wfn_loader`` door, and before it the only wfn_loader pair here was the
#: SHIM — which cannot fail the way a converted consumer can, because the
#: shim IS the bootstrap's own module.  A real consumer that reaches the
#: door through ``ffi._services`` is a different edge and needs its own
#: cell.  ``operator_checks`` is chosen out of the fourteen because its
#: entire module-scope import list is ``dataclasses``, ``typing`` and the
#: door: it needs no FFI ``.so``, no communicator stack and no deck, so
#: this cell RUNS on every machine instead of skipping (the same reasoning
#: that keeps ``bandstructure.bse_setup`` out, below).  The other thirteen
#: converted module-scope consumers ride the same one bootstrap block, so
#: one executing cell falsifies the class; the Si COHSEX fixture run is
#: what covers them end to end on the cluster.
#: The four ``symmetry_maps`` pairs came from that service's own branch,
#: where each was MEASURED through :func:`_bare_run` at the replumb rather
#: than assumed reachable.  The landing merge re-measures them, because the
#: merge itself changed their import lists: every one of the four now also
#: imports the ``wfn_loader`` door, which the symmetry_maps branch never had
#: in the same file.  A pair that stops bare-launching moves to
#: ``_FFI_BLOCKED_CONSUMERS`` with the measurement written down, never
#: silently deleted.
_MODULE_SCOPE_CONSUMERS = (
    ("isdf.core", "distrib_la"),
    ("bse.vq_interp", "distrib_la"),
    ("file_io.wfn_loader", "wfn_loader"),
    ("psp.operator_checks", "wfn_loader"),
    ("centroid.charge_density", "symmetry_maps"),
    ("centroid.pivoted_cholesky", "symmetry_maps"),
    ("psp.get_DFT_mtxels", "symmetry_maps"),
    ("psp.run_sternheimer", "symmetry_maps"),
)

#: Every service the pairs above reach.  The red arm runs once per service:
#: a machine that has ONE of them installed some other way would otherwise
#: make only that service's cells tautological, silently.
_SERVICES = tuple(dict.fromkeys(svc for _mod, svc in _MODULE_SCOPE_CONSUMERS))

#: THE CENSUS: every module under ``src/`` that reaches the ``wfn_loader``
#: door at MODULE scope, as the AST sees it.  ADOPTED 2026-08-07 (step-3
#: adjudication item 4, on Arm B's blind-audit recommendation).
#:
#: WHY A SECOND LIST.  ``_MODULE_SCOPE_CONSUMERS`` above is a SAMPLE — two
#: of these twelve, chosen because their subprocess cell RUNS on every
#: machine instead of skipping.  That sample is what the bare-launch cells
#: can afford; it is not what the bootstrap has to hold for.  The step-3
#: replumb converted a dozen modules in one sweep, and the next one lands
#: a thirteenth: nothing in the subprocess cells notices, because the
#: sample does not grow when the population does.  The structural cell
#: below closes exactly that: it re-derives this list from the AST and
#: fails when the two disagree, so an unlisted consumer is a RED CELL
#: rather than a silent hole with a green suite over it.
#:
#: RELATIVE IMPORTS ARE NOT DOOR EDGES, and the distinction is the whole
#: reason this is derived rather than grepped.  ``src/file_io/__init__.py``
#: says ``from .wfn_loader import WfnLoader`` — level 1, the SHIM in its own
#: package, which needs no bootstrap because it IS the module that runs one.
#: A detector that reads ``node.module`` without ``node.level`` (the step-3
#: prototype did) counts it as a thirteenth door consumer and then demands
#: a bootstrap that would be circular.
_ALL_MODULE_SCOPE_DOOR_CONSUMERS = frozenset({
    "bandstructure.htransform",
    "centroid.charge_density",
    "centroid.kmeans_cli",
    "centroid.pivoted_cholesky",
    "file_io.wfn_loader",
    "gw.gw_jax",
    "gw.kin_ion_io",
    "psp.get_DFT_mtxels",
    "psp.get_dipole_mtxels",
    "psp.operator_checks",
    "psp.orbital_magnetization",
    "psp.run_sternheimer",
})

#: The bootstrap call every one of them must make FIRST.
_BOOTSTRAP = "ensure_on_path"


# ---------------------------------------------------------------------------
# The symmetry_maps BUCKETED census (from svc/symmetry_maps, kept whole)
# ---------------------------------------------------------------------------
# The census above answers "does every module-scope door import have a
# bootstrap above it".  This one answers a different question — "is every
# module-scope consumer ACCOUNTED FOR" — and the difference matters: the
# symmetry_maps replumb put ten module-scope door imports into ``src/`` in
# one go, all of them correctly bootstrapped, and nothing in this file
# noticed they existed.  A consumer that lands in none of the four buckets
# below is a consumer nobody measured, and the four buckets are four
# different coverage stories: bootstrap-covered by a cell here, covered
# only by the cluster fixture run, a shim due for deletion, or a script.
#
# SCOPED TO THE symmetry_maps DOOR, as its own branch scoped it.
# ``_scan_module_scope`` takes the door as an argument, so a twin for any
# other door is one call — but this cell claims nothing about them.

#: Module-scope symmetry_maps consumers that CANNOT get a cell here: each
#: reaches ``runtime.initialize_communicator_stack()`` at import (directly or
#: through a dependency), which requires the FFI ``.so``, so a cell naming
#: one would skip on every machine without it.  MEASURED at the replumb —
#: each went through :func:`_bare_run` and died on the missing library, never
#: on the bootstrap.  The Si COHSEX fixture run covers them on the cluster.
_FFI_BLOCKED_CONSUMERS = (
    "bandstructure.htransform",
    "centroid.kmeans_cli",
    "gw.gw_jax",
    "gw.kin_ion_io",
    "psp.get_dipole_mtxels",
    "psp.orbital_magnetization",
)

#: The three transitional shims — ``src/`` modules whose whole body is a
#: re-export of the door, bound at module scope the way distrib_la's shims
#: bind theirs.  Listed rather than pattern-matched so a FOURTH shim cannot
#: land silently either; they die with the phase-wide shim deletion, and on
#: that day this tuple empties and the census still balances.
_SHIM_CONSUMERS = (
    "centroid.orbit_syms",
    "common.density_symmetry_check",
    "common.symmetry_maps",
)

#: Not importable modules, so ``_bare_run`` cannot parametrize over them —
#: they are entry points, covered the same way the FFI-blocked consumers are.
#: Repo-relative paths, because a script has no dotted name.
_SCRIPT_CONSUMERS = (
    "scripts/checks/sigma_direct_check.py",
)


def _module_scope_statements(tree: ast.Module):
    """Module-scope statements, descending into ``try``/``if`` but not defs.

    A ``def`` or ``class`` body is NOT module scope: a bootstrap call in a
    function runs when the function does, which is after the module-scope
    import has already failed.  Counting one would turn the seven LAZY
    sites this file's header names into false evidence of a bootstrap that
    is not there.
    """
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            continue
        yield from ast.walk(node)


def _door_imports(tree: ast.Module, door: str):
    """``[(lineno, spelling)]`` — module-scope ABSOLUTE imports of ``door``.

    ``level == 0`` only: see ``_ALL_MODULE_SCOPE_DOOR_CONSUMERS`` above.
    """
    hits = []
    for node in _module_scope_statements(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            mod = node.module or ""
        elif isinstance(node, ast.Import):
            mod = ",".join(a.name for a in node.names)
        else:
            continue
        for name in mod.split(","):
            if name == door or name.startswith(f"{door}."):
                hits.append((node.lineno, ast.unparse(node)[:90]))
                break
    return hits


def _bootstrap_lineno(tree: ast.Module):
    """Line of the first module-scope ``ensure_on_path()`` CALL, or None.

    The CALL, not the import of it: ``from ffi import _services`` puts the
    name in scope and does nothing to ``sys.path``.  Several consumers
    import ``_services`` many lines above the call that matters.
    """
    for node in _module_scope_statements(tree):
        if (isinstance(node, ast.Call)
                and ast.unparse(node.func).endswith(_BOOTSTRAP)):
            return node.lineno
    return None


def _scan_tree(src: str, door: str = "wfn_loader"):
    """``(census, unbootstrapped)`` over every ``.py`` under ``src``.

    ``census`` is ``{dotted module: [(lineno, spelling)]}``;
    ``unbootstrapped`` is ``[(dotted, lineno, spelling, why)]`` for every
    door import with no module-scope bootstrap strictly above it.
    """
    census, bad = {}, []
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            hits = _door_imports(tree, door)
            if not hits:
                continue
            dotted = os.path.relpath(path, src)[:-3].replace(os.sep, ".")
            if dotted.endswith(".__init__"):
                dotted = dotted[: -len(".__init__")]
            census[dotted] = hits
            boot = _bootstrap_lineno(tree)
            for lineno, spelling in hits:
                if boot is None:
                    bad.append((dotted, lineno, spelling,
                                f"no module-scope {_BOOTSTRAP}() anywhere"))
                elif boot > lineno:
                    bad.append((dotted, lineno, spelling,
                                f"{_BOOTSTRAP}() is at line {boot}, AFTER "
                                f"the import at {lineno}"))
    return census, bad


def _module_path(dotted: str) -> str:
    """``gw.gw_jax`` → ``src/gw/gw_jax.py``.  RAISES if it is not there.

    A documented name that resolves to no file would otherwise sit in the
    census forever matching nothing — the census would balance and the
    typo would be invisible, which is the failure mode of every hand-kept
    list.  Packages resolve to their ``__init__.py``.
    """
    rel = Path("src", *dotted.split("."))
    for cand in (rel.with_suffix(".py"), rel / "__init__.py"):
        if (_REPO / cand).is_file():
            return cand.as_posix()
    raise AssertionError(
        f"{dotted!r} is named in this file's census but resolves to no file "
        f"under {_REPO / 'src'} — either it moved and the list did not, or "
        f"the name is a typo that has been matching nothing.")


def _scan_module_scope(root: Path, base: Path, door: str) -> set[str]:
    """Every ``.py`` under ``root`` importing ``door`` at MODULE scope.

    Returns ``base``-relative POSIX paths.  MODULE scope means "runs on
    import": a body nested in a ``def``/``async def``/``lambda`` is
    excluded, a body under ``if``/``try``/``with``/``class`` is NOT,
    because those execute when the module is imported and can therefore
    fail a bare launch.  ``from . import`` (level > 0) can never name the
    door and is skipped.

    LAZY IMPORTS ARE OUT OF SCOPE, deliberately and by definition: an
    ``import`` inside a function body cannot fail at import time, so it
    cannot produce the "green suite, red cluster" defect this file is
    about.  It fails on the call path instead, where the end-to-end run
    is what covers it.  (There are 15 such statements reaching
    symmetry_maps as of 2026-08-07; this function must not count them,
    and the red twin measures that it does not.)
    """
    found: set[str] = set()

    def _names(node) -> list[str]:
        if isinstance(node, ast.Import):
            return [a.name for a in node.names]
        if isinstance(node, ast.ImportFrom) and not node.level:
            return [node.module or ""]
        return []

    def _walk(node, in_func: bool) -> bool:
        hit = False
        for child in ast.iter_child_nodes(node):
            nested = in_func or isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if not nested and any(
                    n == door or n.startswith(door + ".")
                    for n in _names(child)):
                hit = True
            hit |= _walk(child, nested)
        return hit

    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _walk(tree, False):
            found.add(path.relative_to(base).as_posix())
    return found


def test_the_module_scope_symmetry_maps_census_is_complete():
    """The tree's module-scope symmetry_maps consumers ARE the four lists.

    Set equality in both directions, so it fails on an addition (an
    eleventh consumer nobody bucketed) and on a deletion (a name kept in
    the census after its file went away) alike.
    """
    documented = (
        {_module_path(m) for m, door in _MODULE_SCOPE_CONSUMERS
         if door == "symmetry_maps"}
        | {_module_path(m) for m in _FFI_BLOCKED_CONSUMERS}
        | {_module_path(m) for m in _SHIM_CONSUMERS}
        | set(_SCRIPT_CONSUMERS))
    found = (_scan_module_scope(_REPO / "src", _REPO, "symmetry_maps")
             | _scan_module_scope(_REPO / "scripts", _REPO, "symmetry_maps"))
    unlisted = sorted(found - documented)
    stale = sorted(documented - found)
    assert not unlisted, (
        f"NEW module-scope `import symmetry_maps` consumer(s) that no list "
        f"in this file carries: {unlisted}.  Decide the bucket before this "
        f"goes green — _MODULE_SCOPE_CONSUMERS if a bare launch can reach "
        f"it (add the cell and MEASURE it), _FFI_BLOCKED_CONSUMERS if it "
        f"pulls in the .so at import, _SHIM_CONSUMERS if it is a "
        f"re-export awaiting the phase-wide deletion, _SCRIPT_CONSUMERS if "
        f"it is an entry point.  An unbucketed consumer is one nobody "
        f"measured, and that is the whole defect class this file is for.")
    assert not stale, (
        f"this file's census names {stale}, which no longer imports "
        f"symmetry_maps at module scope (or no longer exists).  Drop the "
        f"name; a census with dead entries stops being evidence.")


def test_the_module_scope_census_can_fail(tmp_path):
    """RED TWIN.  A synthetic unlisted consumer IS caught; a lazy one is not.

    Three files in a throwaway tree, because the green cell above passes
    on a tree that happens to be tidy and would pass just as green if
    ``_scan_module_scope`` returned the empty set for every input.
    """
    for rel, body in (
            ("src/gw/listed_thing.py", "import symmetry_maps\n"),
            ("src/gw/brand_new_consumer.py",
             "from ffi import _services\n"
             "_services.ensure_on_path()\n"
             "import symmetry_maps\n"),
            ("src/gw/conditional_consumer.py",
             "import os\n"
             "if os.environ.get('X'):\n"
             "    from symmetry_maps import KStarMap\n"),
            ("src/gw/lazy_consumer.py",
             "def build():\n"
             "    import symmetry_maps\n"
             "    return symmetry_maps.KStarMap\n"),
            ("src/gw/innocent.py", "import numpy as np\n"),
    ):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)

    found = _scan_module_scope(tmp_path / "src", tmp_path, "symmetry_maps")

    assert "src/gw/brand_new_consumer.py" in found, (
        "the enumerator missed a module-scope consumer, so the green cell "
        "above proves nothing")
    # A module-scope `if` body still runs on import, so it counts.
    assert "src/gw/conditional_consumer.py" in found
    # ...and a function body does not, which is the docstring's scope
    # claim made executable rather than asserted in English.
    assert "src/gw/lazy_consumer.py" not in found, (
        "the enumerator counted a LAZY import; it would then demand a "
        "bucket for the 15 in-function statements the end-to-end run "
        "covers, and the census would be abandoned as noise")
    assert "src/gw/innocent.py" not in found

    documented = {"src/gw/listed_thing.py"}
    assert found - documented, (
        "the set-difference the green cell asserts on would have been "
        "empty even with an unlisted consumer present")


def _bare_run(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` with PYTHONPATH = <repo>/src, and nothing inherited.

    ``JAX_PLATFORMS=cpu`` keeps it off a shared GPU; the env is otherwise
    stripped to the variables an interpreter needs to start, so a
    ``PYTHONPATH`` that happens to name ``services/`` in the developer's
    shell cannot make this pass.
    """
    env = {
        "PYTHONPATH": _SRC,
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "1",
        # No LORRAX_*, no XLA_FLAGS, and no inherited PYTHONPATH.
    }
    return subprocess.run([sys.executable, "-c", code], env=env, cwd=str(_REPO),
                          capture_output=True, text=True, timeout=600)


@pytest.mark.parametrize("service", _SERVICES)
def test_a_bare_launch_cannot_find_the_service_by_itself(service):
    """THE RED ARM.  ``import <service>`` alone must FAIL in this env.

    If this ever passes, the machine has the service installed some other
    way and every other cell in this file has stopped measuring the
    bootstrap.  Fail loudly rather than let them go quietly tautological.

    Once per SERVICE, not once for the file: a box with an editable
    install of one of them would make only that service's consumer cells
    vacuous, and a single red arm naming the other service would report
    everything as fine.
    """
    r = _bare_run(f"import {service}")
    assert r.returncode != 0, (
        f"`import {service}` SUCCEEDED with PYTHONPATH=<repo>/src only, so "
        f"this environment already exposes services/*/src by some other "
        f"route.  Every {service} cell in this file is now a tautology: "
        f"they would pass with the bootstrap deleted.  Find the route (an "
        f"editable install? a .pth?) before trusting them.")
    assert "ModuleNotFoundError" in r.stderr, r.stderr


@pytest.mark.parametrize("mod,service", _MODULE_SCOPE_CONSUMERS)
def test_a_module_scope_consumer_reaches_the_door_in_a_bare_launch(
        mod, service):
    """…and importing the CONSUMER makes it resolvable, via the bootstrap."""
    r = _bare_run(
        f"import importlib, sys\n"
        f"importlib.import_module({mod!r})\n"
        f"assert {service!r} in sys.modules, 'consumer imported but the "
        f"door did not'\n"
        f"import {service}\n"
        f"print('DOOR', {service}.__file__)\n")
    assert r.returncode == 0, (
        f"{mod} does not import with PYTHONPATH=<repo>/src alone — this is "
        f"the cluster's environment, so this is a RUN failure, not a test "
        f"one.\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")
    assert "DOOR" in r.stdout, r.stdout
    resolved = r.stdout.split("DOOR", 1)[1].strip()
    assert resolved.startswith(str(_REPO / "services" / service)), (
        f"{mod} resolved {service} to {resolved!r}, which is not this "
        f"tree's service — the bootstrap found somebody else's copy")


def test_the_wfn_loader_shim_is_the_same_object_as_the_door_in_a_bare_launch():
    """The SHIM's promise, in the cluster's environment.

    ``src/file_io/wfn_loader.py`` exists so the in-tree spelling ``from
    file_io.wfn_loader import WfnLoader`` keeps working across the
    extraction, and its docstring claims every name it binds is the SAME
    OBJECT the door exports — no second class, no second copy of the
    arithmetic.  That claim is only interesting where the bootstrap is
    load-bearing, so it is asserted HERE rather than in the service suite:
    with PYTHONPATH=<repo>/src and nothing else, ``ensure_on_path`` has to
    have run for the two names to be comparable at all.
    """
    r = _bare_run(
        "import file_io.wfn_loader as shim\n"
        "import wfn_loader as door\n"
        "assert shim.WfnLoader is door.WfnLoader, 'two classes'\n"
        "assert shim.KSpec is door.KSpec\n"
        "assert shim._phdf5_unfold_kernel is door._phdf5_unfold_kernel\n"
        "print('SAME', door.__file__)\n")
    assert r.returncode == 0, (
        f"the wfn_loader shim does not resolve to the door in a bare "
        f"launch.\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")
    assert "SAME" in r.stdout, r.stdout
    resolved = r.stdout.split("SAME", 1)[1].strip()
    assert resolved.startswith(str(_REPO / "services" / "wfn_loader"))


def test_the_bootstrap_is_idempotent_and_appends():
    """``ensure_on_path`` must not shadow an environment that already knows.

    It APPENDS: an editable install or a PYTHONPATH entry keeps winning, so
    a machine that has been taught about ``services/`` does not silently
    switch to the in-tree copy the day a consumer imports the door.

    The expected count is ENUMERATED FROM DISK here, by a different call
    than the one under test (``pathlib.iterdir`` in this process vs
    ``os.listdir`` in the subprocess's ``_services.service_roots``), and
    the two services that must always be there are named.  It was the
    literal ``2`` until 2026-08-07, which is a number every service
    extraction has to come back and bump — four wave-1 branches editing one
    line, i.e. a guaranteed conflict for a fact the filesystem already
    states.  A bare ``len(added) > 0`` would be the tautology to avoid;
    this is neither.
    """
    expected = sorted(p.name for p in (_REPO / "services").iterdir()
                      if (p / "src").is_dir())
    assert {"lxkit", "distrib_la"} <= set(expected), expected
    r = _bare_run(
        "import sys\n"
        "from ffi import _services\n"
        "before = list(sys.path)\n"
        "_services.ensure_on_path()\n"
        "once = list(sys.path)\n"
        "_services.ensure_on_path()\n"
        "twice = list(sys.path)\n"
        "added = [p for p in once if p not in before]\n"
        "assert once == twice, 'not idempotent'\n"
        "assert added, 'added nothing'\n"
        "assert once[:len(before)] == before, 'it PREPENDED; must append'\n"
        "print('ADDED', len(added))\n")
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    # One per ``services/*/src`` that EXISTS — the count is the census, not
    # a constant, so a service that lands without a ``src/`` (docs only, as
    # ``services/wfn_loader`` was for one commit) is visible here as a
    # number that did not move.  ``zeta_loader`` replaced the hardcoded 3
    # with the census itself; the landing merge kept that form, which is the
    # one the comment above was already asking for.
    assert f"ADDED {len(expected)}" in r.stdout, (
        f"the bootstrap added a different number of roots than the "
        f"{len(expected)} services on disk ({expected}):\n{r.stdout}")


# ===========================================================================
#  STRUCTURAL — the POPULATION, not the sample
# ===========================================================================
#  The subprocess cells above are the strongest evidence available (they
#  run the cluster's environment), and they are also the most expensive:
#  one interpreter launch each, and several of the converted consumers
#  cannot be launched on a laptop at all (an FFI ``.so``, a communicator
#  stack, a deck).  So they name FOUR modules out of the population.
#
#  A sample does not grow when the population does.  The step-3 replumb
#  converted twelve modules to module-scope door consumers in one sweep;
#  the next sweep lands a thirteenth, its author copies the import and not
#  the two bootstrap lines above it, and every cell in this file stays
#  green while the cluster run dies on the first import.  That is the same
#  "green suite, red cluster" defect this file was written for, arriving
#  through the one gap the file's own method leaves open.
#
#  These two cells close it structurally: no subprocess, no ``.so``, no
#  deck — pure AST over ``src/`` — so they cover EVERY consumer including
#  the ones that can only be launched on the cluster.
# ===========================================================================

def test_every_module_scope_door_consumer_has_the_bootstrap_before_it():
    """AST census of the door's module-scope importers, three claims.

    NON-EMPTY: a detector that finds nothing agrees with a detector that
    is broken, and the second is the likelier reading after a rename.

    ORDERED: every door import has a module-scope ``ensure_on_path()``
    call on a line STRICTLY ABOVE it.  Order is the whole claim — the
    bootstrap appends to ``sys.path``, so a call below the import runs
    after the ``ImportError`` it was meant to prevent.

    COMPLETE: the census equals ``_ALL_MODULE_SCOPE_DOOR_CONSUMERS``.  A
    thirteenth consumer fails HERE until it is listed, which is the point:
    listing it is the moment somebody reads the bootstrap rule.
    """
    census, unbootstrapped = _scan_tree(_SRC)

    assert census, (
        "the AST census found NO module-scope importer of the wfn_loader "
        "door under src/, and this tree has twelve.  The detector is "
        "broken (a moved src/? a renamed door?), not the tree — and a "
        "silently-empty detector is how this cell would go on passing "
        "after it stopped measuring anything.")

    assert not unbootstrapped, (
        "module-scope door importers with no ensure_on_path() above "
        "them.  PYTHONPATH on the cluster is <repo>/src and nothing "
        "else, so these ImportError on the first real run while the "
        "whole pytest suite stays green (the service conftest puts "
        "services/*/src on the path at collection):\n"
        + "\n".join(f"  {mod} :{lineno} — {why}\n      {spelling}"
                    for mod, lineno, spelling, why in unbootstrapped))

    found = frozenset(census)
    unlisted = sorted(found - _ALL_MODULE_SCOPE_DOOR_CONSUMERS)
    stale = sorted(_ALL_MODULE_SCOPE_DOOR_CONSUMERS - found)
    assert not unlisted and not stale, (
        f"the door-consumer census moved.\n"
        f"  UNLISTED (new module-scope consumers, add them to "
        f"_ALL_MODULE_SCOPE_DOOR_CONSUMERS): {unlisted}\n"
        f"  STALE (listed but no longer importing the door, drop them): "
        f"{stale}\n"
        f"This is not bookkeeping: the list is the record of who depends "
        f"on the bootstrap, and the phase-wide shim deletion reads it.")

    # The sample the subprocess cells run must be drawn from the
    # population this cell measures.  If a pair above named a module that
    # is no longer a module-scope consumer, its cell would be asserting
    # something true of a lazy import and proving nothing.
    sampled = {mod for mod, svc in _MODULE_SCOPE_CONSUMERS
               if svc == "wfn_loader"}
    assert sampled <= found, (
        f"{sorted(sampled - found)} is parametrized above as a "
        f"module-scope wfn_loader consumer but the AST says it does not "
        f"import the door at module scope any more — its bare-launch cell "
        f"has gone vacuous.")


def test_the_census_detector_catches_a_missing_and_a_late_bootstrap(tmp_path):
    """THE RED TWIN for the cell above, on a tree built to be wrong.

    Without it, ``_scan_tree`` returning an empty ``unbootstrapped`` list
    is equally consistent with "every consumer is correct" and "the
    detector cannot see a defect" — and the second is what a refactor of
    ``_bootstrap_lineno`` produces, silently, on a tree where the first is
    also true.  Four synthetic modules, one of each shape that matters:

    * ``good`` — bootstrap, then the import.  Must NOT be flagged.
    * ``missing`` — the import with no bootstrap at all.
    * ``late`` — the bootstrap BELOW the import (the ordering claim; a
      detector that only asked "is ``ensure_on_path`` in this file" passes
      it, and the module still dies on the cluster).
    * ``lazy`` — the bootstrap and the import both inside a function.  Not
      a module-scope consumer, must not appear in the census AT ALL; the
      seven real lazy sites are covered by the end-to-end run instead, and
      a detector that flagged them would make this cell unpassable.
    """
    (tmp_path / "good.py").write_text(
        "from ffi import _services\n"
        "_services.ensure_on_path()\n"
        "from wfn_loader import WfnLoader\n")
    (tmp_path / "missing.py").write_text(
        "from wfn_loader import WfnLoader\n")
    (tmp_path / "late.py").write_text(
        "from wfn_loader import WfnLoader\n"
        "from ffi import _services\n"
        "_services.ensure_on_path()\n")
    (tmp_path / "lazy.py").write_text(
        "def load(p):\n"
        "    from ffi import _services\n"
        "    _services.ensure_on_path()\n"
        "    from wfn_loader import WfnLoader\n"
        "    return WfnLoader(p)\n")
    # Relative — the shim's own package spelling.  Not a door edge.
    (tmp_path / "relative.py").write_text(
        "from .wfn_loader import WfnLoader\n")

    census, bad = _scan_tree(str(tmp_path))

    assert set(census) == {"good", "missing", "late"}, (
        f"census wrong: {sorted(census)}.  'lazy' is a function-scope "
        f"import and 'relative' is level-1 (the shim spelling); neither "
        f"is a module-scope door edge.")
    flagged = {mod: why for mod, _ln, _sp, why in bad}
    assert set(flagged) == {"missing", "late"}, (
        f"the detector flagged {sorted(flagged)}; it must flag exactly "
        f"the two defective modules — flagging 'good' would make the "
        f"green cell above meaningless, and missing 'late' means the "
        f"ORDER claim is not being checked.")
    assert "no module-scope" in flagged["missing"], flagged["missing"]
    assert "AFTER the import" in flagged["late"], flagged["late"]

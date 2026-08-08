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
_MODULE_SCOPE_CONSUMERS = (
    ("isdf.core", "distrib_la"),
    ("bse.vq_interp", "distrib_la"),
    ("file_io.wfn_loader", "wfn_loader"),
    ("psp.operator_checks", "wfn_loader"),
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

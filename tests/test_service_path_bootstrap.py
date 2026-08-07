"""A service door is reachable in a BARE launch, not just under pytest.

THE DEFECT THIS EXISTS TO CATCH is "green suite, red cluster".  Nothing in
the launch chain puts ``services/*/src`` on ``sys.path``: ``lx`` rewrites
the container ``PYTHONPATH`` to exactly ``<checkout>/src``
(``~/bin/lx``, ``retarget_pythonpath``), the Shifter image pip-installs
nothing, and ``tests/harness.run_gw_jax`` — the launcher every end-to-end
regression gate goes through — sets ``PYTHONPATH`` to ``<repo>/src`` and
nothing else.  Under pytest the path is there anyway, because
each service's own ``tests/conftest.py`` inserts it at collection.  So a
module-scope ``from distrib_la import ...`` with no bootstrap would pass
the entire lorrax suite and ``ImportError`` on the first real run.

``ffi._services.ensure_on_path()`` is the bootstrap, and it is transitional
plumbing with an owner decision behind it (see its docstring: pip install -e
services/*, a modulefile PYTHONPATH entry, or a uv workspace — all touch
shared resources).  These cells are what keeps it honest until then.

Each cell launches a SUBPROCESS with ``PYTHONPATH=<repo>/src`` and nothing
else, which is the cluster's environment exactly, and imports a consumer
that reaches the door at MODULE scope.

RED ARM: ``test_a_bare_launch_cannot_find_the_service_by_itself`` shows the
same subprocess FAILING to import the door directly.  Without it these
cells could be passing because the service happens to be installed, and
would say nothing about the bootstrap at all.

PER SERVICE, NOT PER TREE.  The cells were written against ``distrib_la``
and hard-coded its name in five places; the symmetry_maps replumb put TEN
more module-scope door imports into ``src/`` and not one of them was
measured by anything here — the exact failure class this file exists for,
arriving as a second service rather than as a bootstrap-free consumer of
the first.  So the consumer list carries the door it reaches, and the red
arm runs once per door.
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

#: ``(consumer, door)`` — modules that import a service door at MODULE
#: scope, paired with the door they reach.  The LAZY sites (the
#: ``ensure_on_path()`` inside a function body: 7 statements reaching
#: distrib_la, 15 reaching symmetry_maps) cannot fail at import time and
#: are covered by the end-to-end run instead.
#:
#: THE OMISSIONS ARE NAMED, because a list nobody can audit is a list that
#: rots.  ``src/`` holds TEN module-scope ``symmetry_maps`` consumers
#: outside the three transitional shims; the four here are the ones a
#: machine with no FFI library can launch at all.  The other six —
#: ``bandstructure.htransform``, ``centroid.kmeans_cli``, ``gw.gw_jax``,
#: ``gw.kin_ion_io``, ``psp.get_dipole_mtxels``,
#: ``psp.orbital_magnetization`` — reach
#: ``runtime.initialize_communicator_stack()`` at import (directly or
#: through a dependency), which REQUIRES the FFI ``.so``, so a cell naming
#: one would be a skip on every machine without it: the exact shape of
#: coverage that evaporates quietly.  MEASURED, not assumed — each was put
#: through :func:`_bare_run` at the replumb and died on the missing
#: library, never on the bootstrap.  The Si COHSEX fixture run covers them
#: on the cluster, where the bootstrap is what the whole gate depends on.
#:
#: ``scripts/checks/sigma_direct_check.py`` is a script, not an importable
#: module, and is covered the same way; ``bandstructure.bse_setup`` is
#: distrib_la's third module-scope consumer and is absent for the same FFI
#: reason it always was.
_MODULE_SCOPE_CONSUMERS = (
    ("isdf.core", "distrib_la"),
    ("bse.vq_interp", "distrib_la"),
    ("centroid.charge_density", "symmetry_maps"),
    ("centroid.pivoted_cholesky", "symmetry_maps"),
    ("psp.get_DFT_mtxels", "symmetry_maps"),
    ("psp.run_sternheimer", "symmetry_maps"),
)

#: Every door some consumer above reaches.  Derived, not written twice.
_DOORS = tuple(sorted({door for _, door in _MODULE_SCOPE_CONSUMERS}))


# ---------------------------------------------------------------------------
# O1 — the census, so the list above cannot rot in silence
# ---------------------------------------------------------------------------
# Everything the block comment on ``_MODULE_SCOPE_CONSUMERS`` says about the
# ten it does NOT carry is prose, and prose is exactly what rots: the
# symmetry_maps replumb put ten module-scope door imports into ``src/`` in
# one go and nothing in this file noticed, which is the defect this file
# exists for arriving as a second service.  Naming them in a docstring fixed
# the audit once.  It does not fix the eleventh.
#
# So the other ten are STRUCTURES, and
# :func:`test_the_module_scope_symmetry_maps_census_is_complete` AST-walks
# ``src/`` and ``scripts/`` and asserts the found set EQUALS their union.  A
# new module-scope consumer therefore fails this cell until somebody decides
# which of the four buckets it belongs in — and that decision is the point,
# because the buckets are four different coverage stories: bootstrap-covered
# here, covered only by the cluster fixture run, a shim due for deletion, or
# a script.  A consumer that lands in none of them is a consumer nobody
# measured.
#
# SCOPED TO THE symmetry_maps DOOR.  ``_scan_module_scope`` takes the door as
# an argument, so a distrib_la twin is one call — but this cell claims
# nothing about distrib_la, whose own census belongs to its service branch.

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


@pytest.mark.parametrize("door", _DOORS)
def test_a_bare_launch_cannot_find_the_service_by_itself(door):
    """THE RED ARM.  ``import <door>`` alone must FAIL in this env.

    If this ever passes, the machine has the service installed some other
    way and every other cell in this file has stopped measuring the
    bootstrap.  Fail loudly rather than let them go quietly tautological.

    It runs PER DOOR because the answer is per door: a tree could have
    ``distrib_la`` unreachable and ``symmetry_maps`` on a stray ``.pth``,
    and one red arm for both would report the wrong service as honest.
    """
    r = _bare_run(f"import {door}")
    assert r.returncode != 0, (
        f"`import {door}` SUCCEEDED with PYTHONPATH=<repo>/src only, so "
        f"this environment already exposes services/*/src by some other "
        f"route.  Every {door} cell in this file is now a tautology: they "
        f"would pass with the bootstrap deleted.  Find the route (an "
        f"editable install? a .pth?) before trusting them.")
    assert "ModuleNotFoundError" in r.stderr, r.stderr


@pytest.mark.parametrize("mod,door", _MODULE_SCOPE_CONSUMERS)
def test_a_module_scope_consumer_reaches_the_door_in_a_bare_launch(mod, door):
    """…and importing the CONSUMER makes it resolvable, via the bootstrap."""
    r = _bare_run(
        f"import importlib, sys\n"
        f"importlib.import_module({mod!r})\n"
        f"assert {door!r} in sys.modules, 'consumer imported but the "
        f"door did not'\n"
        f"import {door}\n"
        f"print('DOOR', {door}.__file__)\n")
    assert r.returncode == 0, (
        f"{mod} does not import with PYTHONPATH=<repo>/src alone — this is "
        f"the cluster's environment, so this is a RUN failure, not a test "
        f"one.\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")
    assert "DOOR" in r.stdout, r.stdout
    # Consumers print at import (``runtime``'s startup banner, for one), so
    # take the DOOR line, not everything after the marker.
    resolved = r.stdout.split("DOOR", 1)[1].strip().splitlines()[0].strip()
    assert resolved.startswith(str(_REPO / "services" / door)), (
        f"{mod} resolved {door} to {resolved!r}, which is not this "
        f"tree's service — the bootstrap found somebody else's copy")


def test_the_bootstrap_is_idempotent_and_appends():
    """``ensure_on_path`` must not shadow an environment that already knows.

    It APPENDS: an editable install or a PYTHONPATH entry keeps winning, so
    a machine that has been taught about ``services/`` does not silently
    switch to the in-tree copy the day a consumer imports the door.
    """
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
    # lxkit + distrib_la + symmetry_maps.  A COUNT, not a floor: the number
    # is `service_roots()`'s answer, so a service that lands without a
    # `src/` directory — or a stale one that leaves an empty shell behind —
    # shows up here rather than in whatever imports it first.
    assert "ADDED 3" in r.stdout, r.stdout

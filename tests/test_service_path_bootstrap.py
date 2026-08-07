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

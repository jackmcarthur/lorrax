"""The distrib_la door is reachable in a BARE launch, not just under pytest.

THE DEFECT THIS EXISTS TO CATCH is "green suite, red cluster".  Nothing in
the launch chain puts ``services/*/src`` on ``sys.path``: ``lx`` rewrites
the container ``PYTHONPATH`` to exactly ``<checkout>/src``
(``~/bin/lx``, ``retarget_pythonpath``), the Shifter image pip-installs
nothing, and ``tests/harness.run_gw_jax`` — the launcher every end-to-end
regression gate goes through — sets ``PYTHONPATH`` to ``<repo>/src`` and
nothing else.  Under pytest the path is there anyway, because
``services/distrib_la/tests/conftest.py`` inserts it at collection.  So a
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
same subprocess FAILING to import ``distrib_la`` directly.  Without it
these cells could be passing because the service happens to be installed,
and would say nothing about the bootstrap at all.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SRC = str(_REPO / "src")

#: Consumers that import the door at MODULE scope.  The lazy sites (seven
#: of them, all ``ensure_on_path()`` inside the function) cannot fail at
#: import time and are covered by the end-to-end run instead.
#:
#: ``bandstructure.bse_setup`` is the third module-scope consumer and is
#: deliberately NOT here: it runs ``initialize_communicator_stack()`` at
#: import (through ``.htransform``), which REQUIRES an FFI ``.so``, so a
#: cell naming it would be a skip on every machine without one — the exact
#: shape of coverage that evaporates quietly.  The Si COHSEX fixture run
#: covers it on the cluster, where the bootstrap is what the whole gate
#: depends on.
_MODULE_SCOPE_CONSUMERS = ("isdf.core", "bse.vq_interp")


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


def test_a_bare_launch_cannot_find_the_service_by_itself():
    """THE RED ARM.  ``import distrib_la`` alone must FAIL in this env.

    If this ever passes, the machine has the service installed some other
    way and every other cell in this file has stopped measuring the
    bootstrap.  Fail loudly rather than let them go quietly tautological.
    """
    r = _bare_run("import distrib_la")
    assert r.returncode != 0, (
        "`import distrib_la` SUCCEEDED with PYTHONPATH=<repo>/src only, so "
        "this environment already exposes services/*/src by some other "
        "route.  Every other cell in this file is now a tautology: they "
        "would pass with the bootstrap deleted.  Find the route (an "
        "editable install? a .pth?) before trusting them.")
    assert "ModuleNotFoundError" in r.stderr, r.stderr


@pytest.mark.parametrize("mod", _MODULE_SCOPE_CONSUMERS)
def test_a_module_scope_consumer_reaches_the_door_in_a_bare_launch(mod):
    """…and importing the CONSUMER makes it resolvable, via the bootstrap."""
    r = _bare_run(
        f"import importlib, sys\n"
        f"importlib.import_module({mod!r})\n"
        f"assert 'distrib_la' in sys.modules, 'consumer imported but the "
        f"door did not'\n"
        f"import distrib_la\n"
        f"print('DOOR', distrib_la.__file__)\n")
    assert r.returncode == 0, (
        f"{mod} does not import with PYTHONPATH=<repo>/src alone — this is "
        f"the cluster's environment, so this is a RUN failure, not a test "
        f"one.\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")
    assert "DOOR" in r.stdout, r.stdout
    resolved = r.stdout.split("DOOR", 1)[1].strip()
    assert resolved.startswith(str(_REPO / "services" / "distrib_la")), (
        f"{mod} resolved distrib_la to {resolved!r}, which is not this "
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
    # One entry per services/*/src, so this number is a CENSUS of the
    # registered services and moves whenever one lands: lxkit + distrib_la
    # were 2; vcoul made it 3 on 2026-08-07.  Asserting the count rather
    # than merely "added something" is what makes a service that silently
    # failed to register visible here.
    assert "ADDED 3" in r.stdout, r.stdout      # lxkit + distrib_la + vcoul

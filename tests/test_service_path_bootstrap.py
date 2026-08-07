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
_MODULE_SCOPE_CONSUMERS = (
    ("isdf.core", "distrib_la"),
    ("bse.vq_interp", "distrib_la"),
    ("file_io.wfn_loader", "wfn_loader"),
)

#: Every service the pairs above reach.  The red arm runs once per service:
#: a machine that has ONE of them installed some other way would otherwise
#: make only that service's cells tautological, silently.
_SERVICES = tuple(dict.fromkeys(svc for _mod, svc in _MODULE_SCOPE_CONSUMERS))


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
    # One per ``services/*/src`` that EXISTS — the count is the census, not
    # a constant, so a service that lands without a ``src/`` (docs only, as
    # ``services/wfn_loader`` was for one commit) is visible here as a
    # number that did not move.
    assert "ADDED 3" in r.stdout, r.stdout   # lxkit + distrib_la + wfn_loader

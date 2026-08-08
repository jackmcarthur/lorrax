"""WAVE-1 COMPAT SHIM — ``ZetaLoader`` lives in ``services/zeta_loader`` now.

The class moved verbatim to ``services/zeta_loader/src/zeta_loader/
loader.py`` and is reached through the package door, ``import
zeta_loader``.  This module is the old import path kept working so that
lorrax stays green with ZERO call-site changes: ``file_io/__init__.py``,
``bse/vq_interp.py``, ``gw/gw_init.py`` and the ζ half of the test suite
all keep saying ``from file_io.zeta_loader import ZetaLoader``.

DELETION GATE.  Not the replumb of this service alone — the PHASE-WIDE
cleanup after all four wave-1 branches land (wave-1 ruling 2).  Four
services are being extracted against one tree at once; a shim deleted on
one branch is a merge conflict and an ImportError on the other three, so
every wave-1 shim stays until the last branch is in and one commit
removes them together.  Nothing new should be written against this path.

``ffi._services.ensure_on_path()`` is why the import below resolves at
all: nothing in the launch chain knows ``services/`` exists (``lx``
rewrites the container ``PYTHONPATH`` to exactly ``<checkout>/src`` and
the Shifter image pip-installs nothing).  It is transitional plumbing
with an owner decision behind it — see ``src/ffi/_services.py``.
"""
from __future__ import annotations

from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                               # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from zeta_loader import ZetaLoader                          # noqa: E402


__all__ = ["ZetaLoader"]

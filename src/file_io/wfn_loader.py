"""TRANSITIONAL SHIM.  ``WfnLoader`` lives in the ``wfn_loader`` service now.

The module moved to ``services/wfn_loader/src/wfn_loader/loader.py`` and is
reached through the package door, :mod:`wfn_loader`.  This file exists so
that the in-tree call sites spelling ``from file_io.wfn_loader import
WfnLoader`` (and ``from file_io import WfnLoader`` / ``WFNReader``, which
``file_io/__init__.py`` routes through here) keep working across the
extraction commit — the sweep command below is the census, because a
number written here is a number nobody comes back to re-count.  Every
name it binds is the SAME OBJECT the door
exports — ``file_io.wfn_loader.WfnLoader is wfn_loader.WfnLoader`` — so
there is no second class and no second copy of the arithmetic.

DELETION GATE: the PHASE-WIDE cleanup commit, after all four wave-1
service branches have landed.  Not this branch's own follow-up, and that
is coordination ruling 2: consumers of this path exist on the OTHER wave-1
branches, which are being written against ``file_io.wfn_loader`` right
now.  Deleting it here would break them at merge, and re-adding it is how
a shim acquires a second life.  When the gate opens the sweep is::

    grep -rn 'file_io.wfn_loader\\|from file_io import.*WfnLoader' src/ tests/

and every hit becomes ``from wfn_loader import ...``.

``ffi._services.ensure_on_path()`` is the bootstrap that makes the import
below resolve in a bare launch: nothing in the launch chain puts
``services/*/src`` on ``sys.path`` (``lx`` rewrites the container
``PYTHONPATH`` to exactly ``<checkout>/src`` and the Shifter image
pip-installs nothing), so a module-scope ``from wfn_loader import ...``
with no bootstrap would pass the whole pytest suite and ``ImportError`` on
the first real run.  See ``tests/test_service_path_bootstrap.py``.
"""
from ffi import _services

_services.ensure_on_path()

from wfn_loader import (                                 # noqa: E402,F401
    KSpec,
    WfnLoader,
    _bispinor_lift_kernel,
    _get_bispinor_lift_jit,
    _phdf5_unfold_kernel,
    _sharded_zero_proto_fn,
)

__all__ = ["WfnLoader"]

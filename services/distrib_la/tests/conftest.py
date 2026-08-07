"""Put the in-tree ``distrib_la`` (and its lxkit foundation) on the path.

Installing the services registers :mod:`lxkit.testing` through the
``pytest11`` entry point, but this suite must also run from a bare
checkout, so it imports the autouse fixture directly.  ``pytest_plugins``
is not an option: pytest honours it only in the ROOTDIR conftest, and this
suite is collected under the monorepo root.
"""

import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_SERVICES = os.path.dirname(os.path.dirname(_TESTS))

for _svc in ("lxkit", "distrib_la"):
    _src = os.path.join(_SERVICES, _svc, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

from lxkit.testing import gate_state          # noqa: E402,F401  (autouse)

"""Put the in-tree ``lxkit`` on the path and arm its autouse fixture.

Installing lxkit registers :mod:`lxkit.testing` through the ``pytest11``
entry point, but this suite must also run from a bare checkout, so it
imports the fixture directly.  ``pytest_plugins`` is not an option: pytest
honours it only in the ROOTDIR conftest, and this suite is collected under
the monorepo root.
"""

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from lxkit.testing import gate_state          # noqa: E402,F401  (autouse)

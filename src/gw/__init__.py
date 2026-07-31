"""GW/COHSEX driver package."""

import os

# x64 backstop for `import gw.<anything>` from an entry point that did not
# run ``runtime.set_default_env`` first (tests, notebooks, a sibling CLI
# that imports ``gw.vcoul``).  jax reads this at IMPORT time, so a
# setdefault only bites while jax is still unimported.
#
# NOT the canonical setter, and not load-bearing for the drivers.  Under
# ``python -m gw.gw_jax`` Python imports the PACKAGE before the module, so
# this line does run first — but ``gw_jax`` then calls
# ``runtime.bootstrap()`` (which setdefaults the same variable) before its
# own ``import jax``, and follows it with an explicit
# ``jax.config.update("jax_enable_x64", True)``.  Same for
# ``gw.kin_ion_io``.  Deleting this line would therefore change nothing
# for either driver; it is kept for the import paths that have no
# bootstrap at all.  ``runtime.set_default_env`` is the one to edit.
os.environ.setdefault("JAX_ENABLE_X64", "1")

from .gw_config import read_lorrax_input, read_cohsex_input

__all__ = ["read_lorrax_input", "read_cohsex_input"]

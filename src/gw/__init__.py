"""GW/COHSEX driver package."""

import os

# x64 backstop for `import gw.<anything>` from an entry point that did not
# run ``runtime.set_default_env`` first (tests, notebooks, a sibling CLI
# that imports ``gw.vcoul``).  jax reads this at IMPORT time, so a
# setdefault only bites while jax is still unimported.
#
# NOT the canonical setter, and not load-bearing for the drivers.  Under
# ``python -m gw.gw_jax`` Python imports this package before the module, so
# the package must remain JAX-FREE: the driver's runtime initialization owns
# the first JAX import, backend selection and timing.  The lazy compatibility
# exports below keep this boundary while preserving ``from gw import
# read_lorrax_input`` for external callers.  ``runtime.set_default_env`` is
# the canonical setter.
os.environ.setdefault("JAX_ENABLE_X64", "1")

__all__ = ["read_lorrax_input", "read_cohsex_input"]


def __getattr__(name):
    if name in __all__:
        from . import gw_config
        return getattr(gw_config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))

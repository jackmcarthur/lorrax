"""Compatibility seal for bare-source library imports.

Core drivers seal the declared LORRAX/service package set through
``runtime.initialize_communicator_stack()`` before importing JAX or physics.
Some library modules are also imported directly from a checkout whose only
path entry is ``<checkout>/src``.  Keep that supported without retaining a
second service roster or path-precedence policy here: this compatibility door
delegates to the same metadata-derived, stale-source-refusing closure as the
drivers.

Installed LORRAX uses the same call.  In that mode the closure verifies wheel
metadata and package ownership and does not mutate ``sys.path``.
"""

from __future__ import annotations

from runtime import rank0_print as _rank0_print
from runtime.source_closure import ensure_source_closure as _ensure_source_closure

__all__ = ["ensure_on_path"]


def ensure_on_path() -> None:
    """Seal the one declared package closure; retained for library callers."""
    _ensure_source_closure(print_fn=_rank0_print)

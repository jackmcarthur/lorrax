"""Back-compat shim — the dispatcher moved to :mod:`ffi.linalg`.

``ffi.common.dispatch`` grew into the distributed-linalg facade package
``ffi/linalg/`` (resolution + dispatch + backend registry).  Import from
``ffi.linalg`` instead; this module only re-exports the old names so
external scripts keep working.
"""
from ..linalg import EIGH_BACKENDS, dispatch_eigh  # noqa: F401

__all__ = ["dispatch_eigh", "EIGH_BACKENDS"]

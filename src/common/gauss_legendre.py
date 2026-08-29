"""Compatibility door to vcoul's standalone finite-interval rule owner."""
from __future__ import annotations

from ffi import _services

_services.ensure_on_path()

from vcoul import (
    GAUSS_LEGENDRE_INTERVAL_PROVENANCE,
    gauss_legendre_interval,
)  # noqa: E402


__all__ = [
    "GAUSS_LEGENDRE_INTERVAL_PROVENANCE",
    "gauss_legendre_interval",
]

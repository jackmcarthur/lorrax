"""Re-export of ``ffi.contour`` for ``gw.w_isdf`` (lanes MAUD and K3 edit it).

The native door moved to ``ffi/`` (ARCH M1, 2026-09-25).  This stub goes
when ``w_isdf`` imports ``ffi.contour`` directly (ARCH wave 2).
"""
from ffi.contour import TARGET, contour_accumulator  # noqa: F401

__all__ = ["TARGET", "contour_accumulator"]

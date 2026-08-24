"""Canonical application-side layouts for WFN-loader products."""
from __future__ import annotations

from jax.sharding import PartitionSpec as P

__all__ = ["band_sphere_spec"]


def band_sphere_spec() -> P:
    """The ψ(G-flat) layout ``(k, band, spinor, G)``, bands over XY.

    This is the one LORRAX-application definition used when requesting or
    transforming a G-flat WFN payload.  The separately packaged WFN-loader
    service stays independent: callers pass this spec through its public
    ``load(..., sharding=...)`` boundary rather than making the service import
    application code.
    """
    return P(None, ("x", "y"), None, None)

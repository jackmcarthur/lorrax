"""Canonical application-side layouts for WFN-loader products."""
from __future__ import annotations

from jax.sharding import PartitionSpec as P

__all__ = ["PSI_MUN_SPEC", "PSI_NMU_SPEC", "psi_specs", "band_sphere_spec"]


# The two persistent face orientations used by low-memory GW.  Keep these at
# the dependency-light application owner so common-layer producers can name
# the same layouts without importing the GW bundle that stores them.
PSI_NMU_SPEC = P(None, "x", None, "y")  # (nk, n_X, s, mu_Y)
PSI_MUN_SPEC = P(None, None, "x", "y")  # (nk, s, mu_X, n_Y)


def psi_specs(layout: str) -> tuple[P, P]:
    """Return nmu/mun placements with distributed or replicated band axes."""
    if layout == "face":
        return PSI_NMU_SPEC, PSI_MUN_SPEC
    if layout == "axis":
        return P(None, None, None, "y"), P(None, None, "x", None)
    raise ValueError(f"Unknown wavefunction layout {layout!r}")


def band_sphere_spec() -> P:
    """The ψ(G-flat) layout ``(k, band, spinor, G)``, bands over XY.

    This is the one LORRAX-application definition used when requesting or
    transforming a G-flat WFN payload.  The separately packaged WFN-loader
    service stays independent: callers pass this spec through its public
    ``load(..., sharding=...)`` boundary rather than making the service import
    application code.
    """
    return P(None, ("x", "y"), None, None)

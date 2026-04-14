"""psp/charge_density.py — backwards-compatible shim.

build_G_cart moved to dft_operators.py.
SCF/GW density routines archived to psp/archive/charge_density.py.
"""
from psp.dft_operators import build_G_cart  # noqa: F401

# Re-export archived functions for any remaining callers
from psp.archive.charge_density import (  # noqa: F401
    build_density_from_ibz,
    build_core_density,
    compute_grad_rho_sq,
    build_V_xc,
)

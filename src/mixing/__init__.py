"""
Acceleration methods for fixed-point iterations.

This module provides JAX implementations of Anderson, CROP, and related
acceleration methods for SCF/GW self-consistency loops.

Primary use case: accelerating convergence of Σ^{in} → Σ^{out} in GW iterations,
where x ↔ vec(Σ_mnk) and f(x) = Σ^{out}[x] - Σ^{in}[x].
"""

from .acceleration import (
    anderson_acceleration,
    crop,
    crop_anderson,
    rcrop,
    rcrop_anderson,
)

__all__ = [
    "anderson_acceleration",
    "crop",
    "crop_anderson",
    "rcrop",
    "rcrop_anderson",
]


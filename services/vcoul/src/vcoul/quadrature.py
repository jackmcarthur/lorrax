"""Standalone host Gauss--Legendre rules shared through the vcoul door."""
from __future__ import annotations

import functools

import numpy as np


GAUSS_LEGENDRE_INTERVAL_PROVENANCE = (
    "numpy.polynomial.legendre.leggauss/finite_interval_v1"
)


@functools.lru_cache(maxsize=None)
def gauss_legendre_interval(order: int, left: float, right: float):
    """Return immutable float64 nodes/weights mapped from [-1,1] to [a,b]."""
    n, a, b = int(order), float(left), float(right)
    if isinstance(order, (bool, np.bool_)) or n != order or n <= 0:
        raise ValueError(
            "Gauss--Legendre order must be a positive integer; "
            f"got {order!r}")
    if not (np.isfinite(a) and np.isfinite(b) and b > a):
        raise ValueError(
            "Gauss--Legendre interval must satisfy finite left < right; "
            f"got [{a}, {b}]")
    nodes, weights = np.polynomial.legendre.leggauss(n)
    scale = 0.5 * (b - a)
    nodes = np.asarray(a + scale * (nodes + 1.0), dtype=np.float64)
    weights = np.asarray(scale * weights, dtype=np.float64)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


__all__ = [
    "GAUSS_LEGENDRE_INTERVAL_PROVENANCE",
    "gauss_legendre_interval",
]

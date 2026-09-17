"""A certified positive-time rule for ``1/(x+i*height)`` on ``|x|<=span``.

This is a conservative reference construction, not the unproved corrected
one-sided csc proposal in the 2026-09-16 discussion.  Composite Gauss-Legendre
integrates the causal Laplace representation.  Its panel and tail bounds apply
to the continuum in exact arithmetic; floating-point error is audited apart.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class DampedLineRule:
    span: float
    height: float
    tolerance: float
    times: np.ndarray
    weights: np.ndarray
    order: int
    panels: int
    panel_bound: float
    tail_bound: float

    @property
    def bound(self) -> float:
        return self.panel_bound + self.tail_bound

    @property
    def node_count(self) -> int:
        return self.times.size

    def evaluate(self, x, block: int = 256):
        """Evaluate ``sum_j weights[j] exp(i*x*times[j])``."""
        if block < 1:
            raise ValueError("block must be positive")
        xx = np.asarray(x, dtype=float)
        flat = xx.reshape(-1)
        out = np.empty(flat.size, dtype=np.complex128)
        for start in range(0, flat.size, block):
            part = flat[start:start + block]
            out[start:start + block] = np.exp(1j * part[:, None] * self.times) @ self.weights
        return out.reshape(xx.shape)


def _panel_error(span: float, height: float, duration: float, panels: int,
                 order: int) -> float:
    """Sum the Gauss-Legendre derivative remainder over all panels."""
    width = duration / panels
    radius = math.hypot(span, height)
    log_coefficient = (4 * math.lgamma(order + 1)
                       - 3 * math.lgamma(2 * order + 1)
                       - math.log(2 * order + 1))
    log_bound = (log_coefficient + (2 * order + 1) * math.log(width)
                 + 2 * order * math.log(radius)
                 - math.log(-math.expm1(-height * width)))
    if log_bound > 700:
        return math.inf
    return math.exp(log_bound)


def make_rule(span: float, height: float, tolerance: float,
              orders=range(4, 33, 2)) -> DampedLineRule:
    """Select the fewest nodes among prescribed composite Gaussian orders.

    The causal identity is ``1/(x+i*h)=-i integral_0^inf exp((i*x-h)t) dt``.
    Tail and panel budgets are each half the requested absolute error.
    Every node is positive and every oscillatory factor has modulus one.
    """
    span, height, tolerance = float(span), float(height), float(tolerance)
    if not (math.isfinite(span) and span >= 0 and math.isfinite(height)
            and height > 0 and math.isfinite(tolerance) and tolerance > 0):
        raise ValueError("Need finite span>=0, height>0, tolerance>0")
    duration = max(0.0, math.log(2 / (height * tolerance)) / height)
    if duration == 0:
        return DampedLineRule(span, height, tolerance, np.empty(0),
                              np.empty(0, complex), 0, 0, 0.0, 1 / height)
    best = None
    for order in orders:
        if order < 1 or int(order) != order:
            raise ValueError("Gaussian orders must be positive integers")
        low, high = 1, 1
        while _panel_error(span, height, duration, high, order) > tolerance / 2:
            high *= 2
        while low < high:
            mid = (low + high) // 2
            if _panel_error(span, height, duration, mid, order) <= tolerance / 2:
                high = mid
            else:
                low = mid + 1
        candidate = (order * high, order, high)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("At least one Gaussian order is required")
    _, order, panels = best
    width = duration / panels
    from numpy.polynomial.legendre import leggauss
    nodes, gauss_weights = leggauss(order)
    starts = width * np.arange(panels)
    times = (starts[:, None] + width * (nodes[None, :] + 1) / 2).reshape(-1)
    weights = (-1j * width / 2 * gauss_weights[None, :]
               * np.exp(-height * times.reshape(panels, order))).reshape(-1)
    return DampedLineRule(span, height, tolerance, times, weights, order,
                          panels, _panel_error(span, height, duration, panels, order),
                          math.exp(-height * duration) / height)

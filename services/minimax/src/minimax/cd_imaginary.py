"""Frozen compact CD coefficient family; no response, fit, or W owner.

Linear-rational Chebyshev product weights are served from a run-frozen
piecewise Chebyshev table generated in high precision offline. GreenX's
ordinary du weights are the independent family. Both subtract the same
B(u)=A*b**2/(u**2+b**2) and add its exact Lorentzian integral.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CompactCDImaginary:
    nodes_ev: np.ndarray
    greenx_weights_ev: np.ndarray
    linear_table: np.ndarray
    log_edges: np.ndarray
    scale_ev: float

    @property
    def family_names(self):
        return ("linear_rational_32_dynamic_B", "greenx_24_dynamic_B")

    def __post_init__(self):
        if (self.nodes_ev.shape != (56,) or self.nodes_ev[0] != 0.0
                or np.any(self.nodes_ev[1:] <= 0.0)
                or np.unique(self.nodes_ev).size != 56
                or self.greenx_weights_ev.shape != (24,)
                or self.linear_table.shape[2] != 31
                or self.linear_table.shape[0] + 1 != self.log_edges.size
                or not np.isfinite(self.scale_ev) or self.scale_ev <= 0.0):
            raise ValueError("invalid frozen compact CD imaginary table")
        for values in (self.nodes_ev, self.greenx_weights_ev,
                       self.linear_table, self.log_edges):
            if not np.all(np.isfinite(values)):
                raise ValueError("nonfinite compact CD table")

    def _linear(self, node, a):
        return self._linear_table_value(self.linear_table[:, :, node], a)

    def _linear_table_value(self, c, a):
        logarithm = np.log(np.maximum(a / self.scale_ev, np.finfo(float).tiny))
        nonzero = a != 0.0
        if np.any(nonzero & ((logarithm < self.log_edges[0]) |
                            (logarithm > self.log_edges[-1]))):
            raise ValueError("target denominator outside frozen coefficient domain")
        panel = np.clip(np.searchsorted(self.log_edges, logarithm, side="right") - 1,
                        0, self.log_edges.size - 2)
        local = (2 * logarithm - self.log_edges[panel] - self.log_edges[panel + 1]) / (
            self.log_edges[panel + 1] - self.log_edges[panel])
        local = np.where(nonzero, local, 0.0)
        # Clenshaw retains two scalar work arrays; no target*degree tensor.
        b1, b2 = np.zeros_like(a), np.zeros_like(a)
        for degree in range(c.shape[1] - 1, 0, -1):
            b0 = 2 * local * b1 - b2 + c[panel, degree]
            b2, b1 = b1, b0
        value = local * b1 - b2 + c[panel, 0]
        return np.where(nonzero, value, 0.0)

    def coefficient(self, iw, x_signed):
        """Two shared-family rows for one W; coefficients contain no W data."""
        x = np.asarray(x_signed, np.float64)
        a, sign = abs(x), np.sign(x)
        if iw == 0:
            exact = sign * self.scale_ev / (2 * (a + self.scale_ev))
            rows = np.stack((exact.copy(), exact.copy()))
            # Linearity folds all31 reference-B samples into ONE small
            # table before evaluating the target geometry, not31 Clenshaws.
            u_linear = self.nodes_ev[1:32]
            reference = self.scale_ev**2 / (u_linear*u_linear + self.scale_ev**2)
            reference_table = np.einsum("pdi,i->pd", self.linear_table, reference)
            rows[0] -= sign * self._linear_table_value(reference_table, a)
            for u, weight in zip(self.nodes_ev[32:], self.greenx_weights_ev):
                rows[1] -= x * weight / (np.pi * (a*a + u*u)) * self.scale_ev**2 / (
                    u*u + self.scale_ev**2)
            return rows
        rows = np.zeros((2,) + x.shape, np.float64)
        if iw <= 31:
            rows[0] = sign * self._linear(iw - 1, a)
        else:
            u = self.nodes_ev[iw]
            rows[1] = x * self.greenx_weights_ev[iw - 32] / (np.pi * (a*a + u*u))
        return rows

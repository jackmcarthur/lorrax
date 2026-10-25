"""Shared runtime configuration for the minimax-based χ₀ quadratures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MinimaxConfig:
    """Minimax settings of the χ₀ rules: static, imaginary probe and MPA lines.

    One frozen, all-scalar (hashable) record.  ``target_error`` is the
    absolute error of ``1/x`` in Ry⁻¹ (relative for the analytic GN probe
    and the MPA damped lines) and ``max_nodes`` the node cap.  The Σ
    windows do not read this record: their rules come from
    ``sigma_quadrature_eps`` and the broadening through
    ``gw.sigma_box_plan``.
    """

    target_error: float = 1.0e-6
    max_nodes: int = 64
